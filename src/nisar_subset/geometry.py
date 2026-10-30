"""Map an area of interest (AOI) onto the RSLC radar (zero-Doppler) grid.

NISAR RSLC products carry a ``metadata/geolocationGrid``: a coarse 3-D cube
(heightAboveEllipsoid x zeroDopplerTime x slantRange) holding the map
coordinates (``coordinateX``/``coordinateY`` in ``epsg``) of every radar
grid node at each height. That cube is all that is needed to go from map
coordinates to radar coordinates, so no DEM or ISCE3 install is required:

* forward  (time, range, height) -> (x, y): bilinear interpolation on the cube
* inverse  (x, y, height) -> (time, range): piecewise-linear interpolation on a
  Delaunay triangulation of the cube nodes at that height

Terrain height matters: a target at height h appears at a different slant
range than the same lon/lat on the ellipsoid. The caller supplies one or more
heights; the radar window is the union over all of them.
"""
from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field

import h5py
import numpy as np

log = logging.getLogger(__name__)

_EPOCH_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{0,9})?")


def parse_epoch(units) -> np.datetime64:
    """Epoch from a CF units string like 'seconds since 2025-11-06T00:00:00'."""
    if isinstance(units, bytes):
        units = units.decode()
    m = _EPOCH_RE.findall(str(units))
    if len(m) != 1:
        raise ValueError(f"Could not find an epoch in units attribute {units!r}")
    return np.datetime64(m[0].replace(" ", "T"), "ns")


def time_axis_epoch(ds: h5py.Dataset, default: np.datetime64 | None = None) -> np.datetime64:
    units = ds.attrs.get("units")
    if units is None:
        if default is None:
            raise ValueError(f"{ds.name} has no units attribute")
        return default
    return parse_epoch(units)


def seconds_between(a: np.datetime64, b: np.datetime64) -> float:
    """b - a in seconds (float, ns precision)."""
    return (np.datetime64(b, "ns") - np.datetime64(a, "ns")).astype(np.int64) * 1e-9


def align_time_axis(vals: np.ndarray, offset: float, t0: float, t1: float,
                     name: str = "") -> np.ndarray:
    """Express a LUT time axis relative to the swath epoch.

    ISCE3 evaluates RSLC LUTs with their raw time values, i.e. it assumes they
    share the swath epoch. If the axis's units name a different epoch, the
    shifted axis is used only when it overlaps the image and the raw one does
    not (otherwise the units string is treated as inconsistent metadata).
    """
    if offset == 0.0:
        return vals
    overlaps = lambda v: v.size and v.min() <= t1 and v.max() >= t0  # noqa: E731
    if overlaps(vals):
        log.warning("%s: units epoch differs from the swath epoch but raw times "
                    "overlap the image; using raw times (as ISCE3 does)", name)
        return vals
    return vals + offset


@dataclass
class RadarWindow:
    """A rectangular window on the RSLC grid.

    Rows index ``swaths/zeroDopplerTime`` (shared by all frequencies); column
    windows are per frequency because frequency A and B have different range
    sampling. All stops are exclusive.
    """
    row_start: int
    row_stop: int
    cols: dict[str, tuple[int, int]]
    # Physical extent (swath epoch seconds / meters) of the selected samples,
    # used to crop the metadata LUTs consistently for every frequency.
    t_first: float = 0.0
    t_last: float = 0.0
    r_first: float = 0.0
    r_last: float = 0.0
    heights: list[float] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.row_stop - self.row_start

    def width(self, freq: str) -> int:
        a, b = self.cols[freq]
        return b - a

    def __str__(self) -> str:
        cols = ", ".join(f"{f}: [{a}:{b}]" for f, (a, b) in self.cols.items())
        return f"rows [{self.row_start}:{self.row_stop}], cols {{{cols}}}"


class GeolocationGrid:
    """Interpolators over ``metadata/geolocationGrid``."""

    def __init__(self, group: h5py.Group, swath_epoch: np.datetime64,
                 swath_span: tuple[float, float] | None = None):
        self.x_cube = group["coordinateX"][()].astype(np.float64)
        self.y_cube = group["coordinateY"][()].astype(np.float64)
        self.heights = group["heightAboveEllipsoid"][()].astype(np.float64)
        tds = group["zeroDopplerTime"]
        # Express grid times relative to the swath epoch.
        offset = seconds_between(swath_epoch, time_axis_epoch(tds, swath_epoch))
        self.t = tds[()].astype(np.float64)
        if swath_span is not None:
            self.t = align_time_axis(self.t, offset, *swath_span, tds.name)
        else:
            self.t = self.t + offset
        self.r = group["slantRange"][()].astype(np.float64)
        self.epsg = int(group["epsg"][()]) if "epsg" in group else 4326
        for name, ax in (("zeroDopplerTime", self.t), ("slantRange", self.r)):
            if ax.size < 2 or not np.all(np.diff(ax) > 0):
                raise ValueError(f"{group.name}/{name} is not strictly increasing")
        if self.x_cube.shape != (self.heights.size, self.t.size, self.r.size):
            raise ValueError(f"{group.name} cube shape {self.x_cube.shape} does not match its axes")
        self.wrap = False
        if self.epsg == 4326:
            finite = np.isfinite(self.x_cube)
            if finite.any() and np.ptp(self.x_cube[finite]) > 180:
                # Scene crosses the antimeridian: work in [0, 360) longitudes.
                self.wrap = True
                self.x_cube = np.where(self.x_cube < 0, self.x_cube + 360.0, self.x_cube)
        order_h = np.argsort(self.heights)
        self.heights = self.heights[order_h]
        self.x_cube = self.x_cube[order_h]
        self.y_cube = self.y_cube[order_h]
        self._inverse_cache: dict[float, object] = {}

    # -- forward ---------------------------------------------------------
    def plane(self, h: float) -> tuple[np.ndarray, np.ndarray]:
        """X, Y (nt, nr) at height h, linear in height (clamped to the cube)."""
        hs = self.heights
        if len(hs) == 1 or h <= hs[0]:
            k0 = k1 = 0
            w = 0.0
        elif h >= hs[-1]:
            k0 = k1 = len(hs) - 1
            w = 0.0
        else:
            k1 = int(np.searchsorted(hs, h))
            k0 = k1 - 1
            w = (h - hs[k0]) / (hs[k1] - hs[k0])
        x = (1 - w) * self.x_cube[k0] + w * self.x_cube[k1]
        y = (1 - w) * self.y_cube[k0] + w * self.y_cube[k1]
        return x, y

    def forward(self, t, r, h: float) -> tuple[np.ndarray, np.ndarray]:
        """Map coordinates of radar points (t, r) at height h."""
        from scipy.interpolate import RegularGridInterpolator

        x, y = self.plane(h)
        pts = np.column_stack([np.ravel(t), np.ravel(r)])
        kw = dict(method="linear", bounds_error=False, fill_value=None)
        fx = RegularGridInterpolator((self.t, self.r), x, **kw)(pts)
        fy = RegularGridInterpolator((self.t, self.r), y, **kw)(pts)
        if self.wrap:
            fx = np.where(fx > 180, fx - 360, fx)
        return fx.reshape(np.shape(t)), fy.reshape(np.shape(t))

    # -- inverse ---------------------------------------------------------
    def inverse(self, xs, ys, h: float) -> tuple[np.ndarray, np.ndarray]:
        """Radar (t, r) of map points at height h; NaN outside the grid."""
        from scipy.interpolate import LinearNDInterpolator

        key = float(h)
        interp = self._inverse_cache.get(key)
        if interp is None:
            x, y = self.plane(h)
            tt, rr = np.meshgrid(self.t, self.r, indexing="ij")
            ok = np.isfinite(x) & np.isfinite(y)
            interp = LinearNDInterpolator(np.column_stack([x[ok], y[ok]]),
                                          np.column_stack([tt[ok], rr[ok]]))
            self._inverse_cache[key] = interp
        xs = np.asarray(xs, dtype=np.float64)
        if self.wrap:
            xs = np.where(xs < 0, xs + 360.0, xs)
        out = interp(np.column_stack([xs.ravel(), np.ravel(ys)]))
        return out[:, 0], out[:, 1]

    def node_spacing(self) -> float:
        """Median map distance between neighbouring grid nodes (map units)."""
        x, y = self.plane(float(np.median(self.heights)))
        d = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))
        d = d[np.isfinite(d)]
        return float(np.median(d)) if d.size else 0.01

    def footprint(self, h: float):
        """Polygon of the full geolocation grid at height h."""
        from shapely.geometry import Polygon

        x, y = self.plane(h)
        ring = np.concatenate([
            np.column_stack([x[0, :], y[0, :]]),
            np.column_stack([x[:, -1], y[:, -1]]),
            np.column_stack([x[-1, ::-1], y[-1, ::-1]]),
            np.column_stack([x[::-1, 0], y[::-1, 0]]),
        ])
        return Polygon(ring)


def aoi_in_grid_crs(aoi, aoi_epsg: int, grid_epsg: int):
    """Reproject the AOI geometry into the geolocation grid CRS if needed."""
    if aoi_epsg == grid_epsg:
        return aoi
    import shapely
    from pyproj import Transformer

    tr = Transformer.from_crs(aoi_epsg, grid_epsg, always_xy=True)
    return shapely.transform(aoi, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))


def aoi_radar_extent(grid: GeolocationGrid, aoi, heights) -> tuple[float, float, float, float]:
    """Bounding (t_min, t_max, r_min, r_max) of the AOI on the radar grid.

    The AOI boundary is densified at ~1/4 of the geolocation-grid node spacing
    and inverse-mapped; grid nodes that fall inside the AOI are added too, so
    AOIs smaller or larger than one grid cell are both handled.
    """
    import shapely

    step = max(grid.node_spacing() / 4.0, 1e-9)
    geom = aoi
    if grid.wrap:
        geom = shapely.transform(aoi, lambda xy: np.column_stack(
            [np.where(xy[:, 0] < 0, xy[:, 0] + 360.0, xy[:, 0]), xy[:, 1]]))
    coords = shapely.get_coordinates(shapely.segmentize(geom, step))
    if coords.size == 0:
        raise ValueError("AOI geometry is empty")

    ts, rs = [], []
    outside = 0
    for h in heights:
        t, r = grid.inverse(coords[:, 0], coords[:, 1], h)
        ok = np.isfinite(t) & np.isfinite(r)
        outside += int((~ok).sum())
        ts.append(t[ok])
        rs.append(r[ok])
        if geom.area > 0:
            x, y = grid.plane(h)
            inside = shapely.contains_xy(geom, x, y)
            if inside.any():
                tt, rr = np.meshgrid(grid.t, grid.r, indexing="ij")
                ts.append(tt[inside])
                rs.append(rr[inside])

    t_all = np.concatenate(ts) if ts else np.empty(0)
    r_all = np.concatenate(rs) if rs else np.empty(0)
    if t_all.size == 0:
        raise ValueError("The AOI does not intersect the RSLC footprint.")
    if outside:
        warnings.warn("Part of the AOI lies outside the RSLC geolocation grid; "
                      "the subset is clipped to the scene.", stacklevel=3)
    return float(t_all.min()), float(t_all.max()), float(r_all.min()), float(r_all.max())


def index_window(axis_start: float, spacing: float, n: int, lo: float, hi: float,
                 pad: int = 0) -> tuple[int, int]:
    """Index range [i0, i1) of a regular axis covering the interval [lo, hi]."""
    i0 = int(np.floor((lo - axis_start) / spacing + 1e-9)) - pad
    i1 = int(np.ceil((hi - axis_start) / spacing - 1e-9)) + 1 + pad
    return max(i0, 0), min(i1, n)
