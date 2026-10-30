"""Spatially subset a NISAR L1 RSLC product into a smaller, valid RSLC.

The output keeps the full NISAR product layout (every group, dataset,
attribute, datatype, compression filter and dimension scale of the input), so
it can be fed to ISCE3 / the NISAR workflows exactly like a full product.

What changes:

``swaths/``
    The image layers, ``zeroDopplerTime``, every ``frequency*/slantRange``,
    per-line datasets (``validSamplesSubSwath*``, masks) are cut to the radar
    window. ``validSamplesSubSwath*`` column indices are shifted into the new
    grid. Image statistics attributes (``min_real_value`` ...) are recomputed,
    and ``sceneCenterAlongTrackSpacing`` / ``sceneCenterGroundRangeSpacing``
    are re-evaluated at the centre of the new scene.
``metadata/`` LUTs
    Groups with their own ``zeroDopplerTime`` / ``slantRange`` axes
    (``geolocationGrid``, ``calibrationInformation``,
    ``processingInformation/parameters``) are cropped to the window plus a
    margin of LUT nodes, so every LUT still covers the whole subset.
    ``orbit`` and ``attitude`` are kept in full.
``identification/``
    ``zeroDopplerStartTime``, ``zeroDopplerEndTime``, ``boundingPolygon``,
    ``isFullFrame`` (+ coverage attribute) and, if polarizations or
    frequencies were dropped, ``listOfFrequencies`` / ``listOfPolarizations``.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np

from . import __version__
from .geometry import (GeolocationGrid, RadarWindow, align_time_axis, aoi_in_grid_crs,
                       aoi_radar_extent, index_window, seconds_between, time_axis_epoch)
from .h5utils import (copy_attrs, copy_dataset, copy_dimension_scales, create_dataset_like,
                      create_group_like, read_string)
from .remote import looks_like_granule_id, open_granule

log = logging.getLogger(__name__)

POLARIZATIONS = {"HH", "HV", "VH", "VV", "RH", "RV", "LH", "LV"}

# Metadata subtrees whose LUTs live on their own (time, range) grids.
LUT_SUBTREES = ("metadata/geolocationGrid", "metadata/calibrationInformation",
                "metadata/processingInformation/parameters")

# 1-D datasets that sit next to LUT axes but are not functions of them.
_NOT_AXIS_FUNCTIONS = {"rangeChirpWeighting", "azimuthChirpWeighting"}

_FREQ_RE = re.compile(r"^frequency([A-Z])$")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_aoi(aoi):
    """Return a shapely geometry from WKT, GeoJSON, a file path or a bbox.

    Accepted: a WKT string; a GeoJSON string or dict (Feature,
    FeatureCollection or geometry); a path to a ``.wkt``/``.geojson``/``.json``
    file; a ``(min_x, min_y, max_x, max_y)`` bounding box; a shapely geometry.
    """
    import shapely
    from shapely.geometry import box
    from shapely.geometry.base import BaseGeometry

    if isinstance(aoi, BaseGeometry):
        return aoi
    if isinstance(aoi, (tuple, list)) and len(aoi) == 4 and all(np.isscalar(v) for v in aoi):
        return box(*map(float, aoi))
    if isinstance(aoi, dict):
        return _geojson_to_geometry(aoi)
    if isinstance(aoi, (str, os.PathLike)):
        text = os.fspath(aoi)
        if len(text) < 4096 and os.path.isfile(text):
            text = Path(text).read_text()
        text = text.strip()
        if text.startswith("{"):
            return _geojson_to_geometry(json.loads(text))
        return shapely.from_wkt(text)
    raise TypeError(f"Unsupported AOI type: {type(aoi)!r}")


def _geojson_to_geometry(obj: dict):
    from shapely.geometry import shape
    from shapely.ops import unary_union

    kind = obj.get("type")
    if kind == "FeatureCollection":
        return unary_union([shape(f["geometry"]) for f in obj["features"]])
    if kind == "Feature":
        return shape(obj["geometry"])
    return shape(obj)


@contextlib.contextmanager
def open_source(source, **remote_kw):
    """Yield ``(h5py.File, label)`` for a path, open h5py.File or granule ID."""
    if isinstance(source, h5py.File):
        yield source, source.filename
    elif isinstance(source, (str, os.PathLike)) and os.path.exists(os.fspath(source)):
        with h5py.File(source, "r") as f:
            yield f, os.fspath(source)
    elif isinstance(source, str) and looks_like_granule_id(source):
        with open_granule(source, **remote_kw) as (f, label):
            yield f, label
    else:
        raise FileNotFoundError(f"{source!r} is neither an existing file nor a NISAR granule ID")


# ---------------------------------------------------------------------------
# Read-only view of the input product
# ---------------------------------------------------------------------------
def _decode_list(ds) -> list[str]:
    v = ds[()]
    v = np.atleast_1d(v)
    out = []
    for item in v:
        s = item.decode() if isinstance(item, bytes) else str(item)
        s = s.strip().strip("\x00")
        if s:
            out.append(s)
    return out


class RslcProduct:
    """Locations and grid parameters of an open RSLC file."""

    def __init__(self, f: h5py.File):
        self.file = f
        sci = f["science"]
        band = next((b for b in ("LSAR", "SSAR") if b in sci), None)
        if band is None:
            raise ValueError("Not a NISAR product: no /science/LSAR or /science/SSAR group")
        self.root = sci[band]
        # The spec says RSLC; early sample products used SLC (as ISCE3 allows).
        name = next((n for n in ("RSLC", "SLC") if n in self.root), None)
        if name is None:
            raise ValueError(f"No RSLC product group under {self.root.name}")
        self.product = self.root[name]
        self.identification = self.root["identification"]
        self.swaths = self.product["swaths"]
        self.metadata = self.product.get("metadata")

        zdt = self.swaths["zeroDopplerTime"]
        self.t = zdt[()].astype(np.float64)
        self.epoch = time_axis_epoch(zdt)
        if "zeroDopplerTimeSpacing" in self.swaths:
            self.dt = float(self.swaths["zeroDopplerTimeSpacing"][()])
        else:
            self.dt = float(np.median(np.diff(self.t)))

        freq_groups = sorted(m.group(1) for k in self.swaths if (m := _FREQ_RE.match(k)))
        listed = []
        if "listOfFrequencies" in self.identification:
            listed = [x for x in _decode_list(self.identification["listOfFrequencies"])
                      if x in freq_groups]
        self.frequencies = listed or freq_groups

        self.ranges: dict[str, np.ndarray] = {}
        self.dr: dict[str, float] = {}
        self.pols: dict[str, list[str]] = {}
        for fr in self.frequencies:
            g = self.swaths[f"frequency{fr}"]
            r = g["slantRange"][()].astype(np.float64)
            self.ranges[fr] = r
            self.dr[fr] = (float(g["slantRangeSpacing"][()]) if "slantRangeSpacing" in g
                           else float(np.median(np.diff(r))))
            if "listOfPolarizations" in g:
                self.pols[fr] = [p.upper() for p in _decode_list(g["listOfPolarizations"])]
            else:
                self.pols[fr] = [k for k in g if k.upper() in POLARIZATIONS]

    @property
    def shape(self) -> dict[str, tuple[int, int]]:
        return {fr: (self.t.size, self.ranges[fr].size) for fr in self.frequencies}

    @property
    def path(self) -> str:
        return self.product.name

    def geolocation_grid(self) -> GeolocationGrid | None:
        if self.metadata is not None and "geolocationGrid" in self.metadata:
            g = self.metadata["geolocationGrid"]
            if "coordinateX" in g and "coordinateY" in g:
                try:
                    return GeolocationGrid(g, self.epoch, (self.t[0], self.t[-1]))
                except ValueError as exc:
                    warnings.warn(f"Unusable geolocation grid ({exc}); bounding polygon and "
                                  "scene-centre spacings will not be updated", stacklevel=2)
        return None

    def reference_height(self, t0: float | None = None, t1: float | None = None) -> float:
        """Mean ``referenceTerrainHeight`` (m) over [t0, t1], or 0 if unavailable."""
        try:
            params = self.metadata["processingInformation/parameters"]
            h = params["referenceTerrainHeight"][()].astype(np.float64)
            tds = params["zeroDopplerTime"]
            t = tds[()] + seconds_between(self.epoch, time_axis_epoch(tds, self.epoch))
        except (KeyError, TypeError, ValueError):
            return 0.0
        if h.shape != t.shape:
            return 0.0
        sel = np.isfinite(h)
        if t0 is not None and t1 is not None:
            in_win = sel & (t >= t0) & (t <= t1)
            if in_win.any():
                sel = in_win
        return float(h[sel].mean()) if sel.any() else 0.0


# ---------------------------------------------------------------------------
# Window computation
# ---------------------------------------------------------------------------
def compute_window(prod: RslcProduct, aoi=None, *, aoi_epsg: int = 4326,
                   rows: Sequence[int] | None = None, cols: Sequence[int] | None = None,
                   frequencies: Sequence[str] | None = None, pad: int = 0,
                   heights: Iterable[float] | None = None) -> RadarWindow:
    """Radar window covering `aoi` (or explicit `rows`/`cols` indices).

    `cols` are indices on the first selected frequency; the other frequency's
    columns are derived from the same slant-range interval.
    """
    freqs = list(frequencies or prod.frequencies)
    n_az = prod.t.size
    ref = freqs[0]
    hs: list[float] = []

    if aoi is None and rows is None and cols is None:
        raise ValueError("Provide an AOI or a rows/cols window")

    if aoi is not None:
        geo = prod.geolocation_grid()
        if geo is None:
            raise ValueError("The product has no metadata/geolocationGrid; use rows/cols instead")
        geom = aoi_in_grid_crs(load_aoi(aoi), aoi_epsg, geo.epsg)
        if heights is not None:
            hs = [float(h) for h in heights]
            tmin, tmax, rmin, rmax = aoi_radar_extent(geo, geom, hs)
        else:
            # Use the processor's reference terrain height: first over the whole
            # scene, then re-evaluated over the AOI's own lines.
            hs = [prod.reference_height()]
            tmin, tmax, rmin, rmax = aoi_radar_extent(geo, geom, hs)
            h2 = prod.reference_height(tmin, tmax)
            if abs(h2 - hs[0]) > 1.0:
                hs = [h2]
                tmin, tmax, rmin, rmax = aoi_radar_extent(geo, geom, hs)
        i0, i1 = index_window(prod.t[0], prod.dt, n_az, tmin, tmax, pad)
        col_windows = {fr: index_window(prod.ranges[fr][0], prod.dr[fr], prod.ranges[fr].size,
                                        rmin, rmax, pad) for fr in freqs}
    else:
        i0, i1 = (0, n_az) if rows is None else (int(rows[0]), int(rows[1]))
        i0, i1 = max(i0 - pad, 0), min(i1 + pad, n_az)
        nr = prod.ranges[ref].size
        c0, c1 = (0, nr) if cols is None else (int(cols[0]), int(cols[1]))
        c0, c1 = max(c0 - pad, 0), min(c1 + pad, nr)
        rmin, rmax = prod.ranges[ref][c0], prod.ranges[ref][max(c1 - 1, c0)]
        col_windows = {ref: (c0, c1)}
        for fr in freqs[1:]:
            col_windows[fr] = index_window(prod.ranges[fr][0], prod.dr[fr],
                                           prod.ranges[fr].size, rmin, rmax, 0)

    if i1 - i0 <= 0 or any(b - a <= 0 for a, b in col_windows.values()):
        raise ValueError("The requested window does not overlap the RSLC image "
                         f"(rows [{i0}:{i1}], cols {col_windows})")

    r_first = min(prod.ranges[fr][a] for fr, (a, b) in col_windows.items())
    r_last = max(prod.ranges[fr][b - 1] for fr, (a, b) in col_windows.items())
    return RadarWindow(i0, i1, col_windows, t_first=float(prod.t[i0]), t_last=float(prod.t[i1 - 1]),
                       r_first=float(r_first), r_last=float(r_last), heights=hs)


# ---------------------------------------------------------------------------
# Copy planning
# ---------------------------------------------------------------------------
@dataclass
class _Axis:
    path: str
    group: str
    n: int
    sl: slice


def _lut_slice(axis: np.ndarray, lo: float, hi: float, margin: int) -> slice:
    """Index slice of a monotonic LUT axis covering [lo, hi] plus `margin` nodes."""
    n = axis.size
    if n < 2:
        return slice(0, n)
    flip = axis[-1] < axis[0]
    a = axis[::-1] if flip else axis
    if not np.all(np.diff(a) > 0):
        return slice(0, n)
    i0 = int(np.searchsorted(a, lo, side="right")) - 1 - margin
    i1 = int(np.searchsorted(a, hi, side="left")) + 1 + margin
    i0, i1 = max(i0, 0), min(i1, n)
    while i1 - i0 < 2:  # keep at least two nodes for interpolation
        i0, i1 = max(i0 - 1, 0), min(i1 + 1, n)
    if flip:
        i0, i1 = n - i1, n - i0
    return slice(i0, i1)


class _Planner:
    """Decides, for every object of the input, what goes into the output."""

    def __init__(self, prod: RslcProduct, win: RadarWindow, freqs: list[str],
                 pols: dict[str, list[str]], crop_metadata: bool, lut_margin: int):
        self.prod = prod
        self.win = win
        self.freqs = freqs
        self.pols = pols
        self.crop_metadata = crop_metadata
        self.lut_margin = lut_margin
        self.rows = slice(win.row_start, win.row_stop)
        self.swaths = prod.swaths.name
        self.lut_roots = [f"{prod.path}/{p}" for p in LUT_SUBTREES]
        self._axis_cache: dict[str, _Axis | None] = {}
        self.all_pols = set().union(*map(set, prod.pols.values())) if prod.pols else set()
        self.kept_pols = set().union(*map(set, pols.values())) if pols else set()

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _frequency_of(path: str) -> str | None:
        for part in reversed(path.split("/")):
            m = _FREQ_RE.match(part)
            if m:
                return m.group(1)
        return None

    def _under(self, path: str, root: str) -> bool:
        return path == root or path.startswith(root + "/")

    def skip_group(self, grp: h5py.Group) -> bool:
        path = grp.name
        if not self._under(path, self.prod.path):
            return False
        name = path.rsplit("/", 1)[-1]
        m = _FREQ_RE.match(name)
        if m and m.group(1) in self.prod.frequencies and m.group(1) not in self.freqs:
            return True
        return self._dropped_pol(path, name)

    def _dropped_pol(self, path: str, name: str) -> bool:
        if name.upper() not in POLARIZATIONS or name.upper() not in self.all_pols:
            return False
        fr = self._frequency_of(path)
        if fr is not None and fr in self.pols:
            return name.upper() not in self.pols[fr]
        return name.upper() not in self.kept_pols

    def _axis(self, group: h5py.Group, name: str) -> _Axis | None:
        """Axis `name` of the nearest ancestor (within a LUT subtree) defining it."""
        root = next((r for r in self.lut_roots if self._under(group.name, r)), None)
        if root is None:
            return None
        g = group
        while True:
            key = f"{g.name}/{name}"
            if key in self._axis_cache:
                return self._axis_cache[key]
            obj = g.get(name)
            if isinstance(obj, h5py.Dataset) and obj.ndim == 1:
                vals = obj[()].astype(np.float64)
                if name == "zeroDopplerTime":
                    try:
                        epoch = time_axis_epoch(obj, self.prod.epoch)
                    except ValueError:
                        epoch = None
                    if epoch is None:
                        # Unknown reference epoch: cannot place the LUT in time,
                        # so keep this axis (and everything on it) in full.
                        log.warning("Keeping %s uncropped (unparseable time units)", obj.name)
                        sl = slice(0, vals.size)
                    else:
                        vals = align_time_axis(vals, seconds_between(self.prod.epoch, epoch),
                                                self.prod.t[0], self.prod.t[-1], obj.name)
                        sl = _lut_slice(vals, self.win.t_first, self.win.t_last, self.lut_margin)
                else:
                    sl = _lut_slice(vals, self.win.r_first, self.win.r_last, self.lut_margin)
                ax = _Axis(obj.name, g.name, vals.size, sl)
                self._axis_cache[key] = ax
                return ax
            if g.name == root:
                return None
            g = g.parent

    # -- main decision ---------------------------------------------------
    def plan(self, ds: h5py.Dataset):
        """Return (kind, payload) for a dataset.

        kinds: "skip", "copy", "slice" (payload: selection), "image"
        (selection; statistics recomputed), "valid" (selection, col0, width).
        """
        path = ds.name
        parent, name = path.rsplit("/", 1)
        if self._under(path, self.prod.path) and self._dropped_pol(path, name):
            return "skip", None
        if self._under(path, self.swaths):
            return self._plan_swath(ds, parent, name)
        if self.crop_metadata and any(self._under(path, r) for r in self.lut_roots):
            return self._plan_lut(ds, parent, name)
        return "copy", None

    def _plan_swath(self, ds, parent, name):
        n_az = self.prod.t.size
        full = lambda k: (slice(None),) * k  # noqa: E731
        if parent == self.swaths:
            if name == "zeroDopplerTime" or (ds.ndim >= 1 and ds.shape[0] == n_az):
                return "slice", (self.rows,) + full(ds.ndim - 1)
            return "copy", None
        m = _FREQ_RE.match(parent.rsplit("/", 1)[-1])
        if not m or parent.rsplit("/", 1)[0] != self.swaths or m.group(1) not in self.win.cols:
            return "copy", None
        fr = m.group(1)
        c0, c1 = self.win.cols[fr]
        cs = slice(c0, c1)
        nr = self.prod.ranges[fr].size
        if name == "slantRange":
            return "slice", (cs,)
        if name.upper() in self.prod.pols.get(fr, []) and ds.shape == (n_az, nr):
            return "image", (self.rows, cs)
        if name.startswith("validSamples") and ds.shape == (n_az, 2):
            return "valid", ((self.rows, slice(0, 2)), c0, c1 - c0)
        if ds.ndim == 2 and ds.shape == (n_az, nr):
            return "slice", (self.rows, cs)
        if ds.ndim >= 1 and ds.shape[0] == n_az:
            return "slice", (self.rows,) + full(ds.ndim - 1)
        if ds.ndim == 1 and ds.shape[0] == nr:
            return "slice", (cs,)
        return "copy", None

    def _plan_lut(self, ds, parent, name):
        grp = ds.parent
        tax = self._axis(grp, "zeroDopplerTime")
        rax = self._axis(grp, "slantRange")
        if name == "zeroDopplerTime" and tax is not None and tax.path == ds.name:
            return "slice", (tax.sl,)
        if name == "slantRange" and rax is not None and rax.path == ds.name:
            return "slice", (rax.sl,)
        if ds.ndim >= 2 and tax is not None and rax is not None and ds.shape[-2:] == (tax.n, rax.n):
            return "slice", (slice(None),) * (ds.ndim - 2) + (tax.sl, rax.sl)
        if ds.ndim == 1 and name not in _NOT_AXIS_FUNCTIONS:
            t_match = tax is not None and tax.group == parent and ds.shape[0] == tax.n
            r_match = rax is not None and rax.group == parent and ds.shape[0] == rax.n
            if t_match and r_match:
                log.warning("Ambiguous axis for %s; copied unchanged", ds.name)
                return "copy", None
            if t_match:
                return "slice", (tax.sl,)
            if r_match:
                return "slice", (rax.sl,)
        return "copy", None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
class _RunningStats:
    """Streaming min / max / mean / sample std (Chan et al. merge)."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = np.inf
        self.max = -np.inf

    def update(self, x: np.ndarray):
        x = x[np.isfinite(x)].astype(np.float64)
        if x.size == 0:
            return
        nb, mb = x.size, float(x.mean())
        m2b = float(((x - mb) ** 2).sum())
        n = self.n + nb
        d = mb - self.mean
        self.mean += d * nb / n
        self.m2 += m2b + d * d * self.n * nb / n
        self.n = n
        self.min = min(self.min, float(x.min()))
        self.max = max(self.max, float(x.max()))

    def value(self, kind: str) -> float:
        if self.n == 0:
            return float("nan")
        if kind == "std":
            return float(np.sqrt(self.m2 / (self.n - 1))) if self.n > 1 else 0.0
        return {"min": self.min, "max": self.max, "mean": self.mean}[kind]


def _split_complex(block: np.ndarray):
    if block.dtype.names:
        names = block.dtype.names
        re_, im_ = ("r", "i") if {"r", "i"} <= set(names) else names[:2]
        return block[re_], block[im_]
    return block.real, block.imag


def _stat_kind(attr: str):
    a = attr.lower()
    comp = "real" if "real" in a else ("imag" if "imag" in a else None)
    if comp is None:
        return None
    if "std" in a or "deviation" in a:
        return comp, "std"
    for k in ("min", "max", "mean"):
        if k in a:
            return comp, k
    return None


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
@dataclass
class SubsetResult:
    output: Path
    window: RadarWindow
    frequencies: list[str]
    polarizations: dict[str, list[str]]
    bounding_polygon: str | None = None
    shapes: dict[str, tuple[int, int]] = field(default_factory=dict)

    def __str__(self) -> str:
        shp = ", ".join(f"{f}: {s[0]} x {s[1]}" for f, s in self.shapes.items())
        return f"{self.output} ({shp}; window {self.window})"


def _copy_tree(src: h5py.Group, dst: h5py.Group, planner: _Planner, stats_out: dict,
               overrides: dict):
    for name in src:
        link = src.get(name, getlink=True)
        if isinstance(link, h5py.SoftLink):
            dst[name] = h5py.SoftLink(link.path)
            continue
        if isinstance(link, h5py.ExternalLink):
            dst[name] = h5py.ExternalLink(link.filename, link.path)
            continue
        obj = src[name]
        if isinstance(obj, h5py.Group):
            if planner.skip_group(obj):
                log.info("Dropping %s", obj.name)
                continue
            _copy_tree(obj, create_group_like(obj, dst, name), planner, stats_out, overrides)
            continue
        if not isinstance(obj, h5py.Dataset):  # committed datatype
            dst[name] = obj
            continue

        if obj.name in overrides:
            _write_override(obj, dst, name, overrides[obj.name])
            continue
        kind, payload = planner.plan(obj)
        if kind == "skip":
            log.info("Dropping %s", obj.name)
        elif kind == "copy":
            copy_dataset(obj, dst, name)
        elif kind == "slice":
            copy_dataset(obj, dst, name, sel=payload)
        elif kind == "valid":
            sel, c0, width = payload

            def shift(block, _rows, c0=c0, width=width):
                b = np.clip(block.astype(np.int64) - c0, 0, width)
                b[b[:, 0] >= b[:, 1]] = 0  # line without valid samples
                return b.astype(block.dtype)

            copy_dataset(obj, dst, name, sel=sel, transform=shift)
        elif kind == "image":
            log.info("Copying %s %s", obj.name, payload)
            re_stats, im_stats = _RunningStats(), _RunningStats()

            def acc(block, re_stats=re_stats, im_stats=im_stats):
                r, i = _split_complex(block)
                re_stats.update(np.asarray(r))
                im_stats.update(np.asarray(i))

            copy_dataset(obj, dst, name, sel=payload, on_block=acc)
            stats_out[obj.name] = {"real": re_stats, "imag": im_stats}
        else:  # pragma: no cover
            raise RuntimeError(kind)


def _format_time(t_seconds: float, epoch: np.datetime64, template: str) -> str:
    """UTC string for epoch + t, with as many fractional digits as `template`."""
    m = re.search(r"\.(\d+)", template)
    # Without a fractional part in the template, use nanoseconds (the format
    # ISCE3's DateTime.isoformat() writes) rather than lose precision.
    digits = len(m.group(1)) if m else 9
    unit = "s" if digits == 0 else "ms" if digits <= 3 else "us" if digits <= 6 else "ns"
    value = epoch + np.timedelta64(int(round(t_seconds * 1e9)), "ns")
    s = np.datetime_as_string(value, unit=unit)
    if digits not in (0, 3, 6, 9):  # trim/pad to the template's precision
        whole, frac = s.split(".")
        s = f"{whole}.{frac[:digits].ljust(digits, '0')}"
    if " " in template and "T" not in template:
        s = s.replace("T", " ")
    return s


def _window_polygon(geo: GeolocationGrid, win: RadarWindow, h: float, n: int = 16):
    from shapely.geometry import Polygon

    ts = np.linspace(win.t_first, win.t_last, n)
    rs = np.linspace(win.r_first, win.r_last, n)
    t = np.concatenate([ts, np.full(n, win.t_last), ts[::-1], np.full(n, win.t_first)])
    r = np.concatenate([np.full(n, win.r_first), rs, np.full(n, win.r_last), rs[::-1]])
    x, y = geo.forward(t, r, h)
    return Polygon(np.column_stack([x, y]))


def _ground_spacing(geo: GeolocationGrid, t: float, r: float, h: float,
                    dt: float, dr: float) -> tuple[float, float]:
    """(along-track, ground-range) metres per pixel at radar point (t, r)."""
    from pyproj import Geod, Transformer

    k = 50.0  # finite difference over k pixels
    tt = np.array([t, t + k * dt, t])
    rr = np.array([r, r, r + k * dr])
    x, y = geo.forward(tt, rr, h)
    if geo.epsg != 4326:
        x, y = Transformer.from_crs(geo.epsg, 4326, always_xy=True).transform(x, y)
    g = Geod(ellps="WGS84")
    _, _, d_az = g.inv(x[0], y[0], x[1], y[1])
    _, _, d_rg = g.inv(x[0], y[0], x[2], y[2])
    return d_az / k, d_rg / k


def _string_like(src: h5py.Dataset, values) -> np.ndarray:
    """`values` (str or list of str) as an array matching src's string type."""
    scalar = isinstance(values, str)
    items = [values] if scalar else list(values)
    dt = src.dtype
    vlen = h5py.check_dtype(vlen=dt)
    if vlen is not None:
        conv = (lambda v: v) if vlen is str else (lambda v: v.encode())
        arr = np.empty(len(items), dtype=dt)
        arr[:] = [conv(v) for v in items]
    else:
        n = max([dt.itemsize if dt.kind == "S" else 1] + [len(v.encode()) for v in items])
        arr = np.array([v.encode() for v in items], dtype=f"S{n}")
    return arr.reshape(()) if scalar and src.shape == () else arr


def _write_override(src: h5py.Dataset, parent: h5py.Group, name: str, value: np.ndarray):
    """Create parent/name holding `value`, mirroring src as closely as possible."""
    value = np.asarray(value)
    if value.dtype == src.dtype and value.shape == src.shape:
        ds = create_dataset_like(src, parent, name)  # identical type and layout
        if value.shape == ():
            ds[()] = value
        else:
            ds[...] = value
        return ds
    if src.shape == () or value.shape != src.shape and src.chunks is None:
        ds = parent.create_dataset(name, data=value)
    else:
        ds = parent.create_dataset(name, data=value, chunks=True if src.chunks else None)
    copy_attrs(src, ds)
    return ds


@dataclass
class _Updates:
    """Metadata changes worked out before the copy starts."""
    datasets: dict = field(default_factory=dict)   # path -> new value (written during copy)
    attrs: dict = field(default_factory=dict)      # path -> {attr: value} (after copy)
    wkt: str | None = None
    h_ref: float = 0.0


def _plan_updates(prod: RslcProduct, win: RadarWindow, freqs: list[str],
                  pols: dict[str, list[str]]) -> _Updates:
    up = _Updates()
    ident = prod.identification
    geo = prod.geolocation_grid()
    up.h_ref = win.heights[0] if win.heights else prod.reference_height(win.t_first, win.t_last)

    # Start / end times of the new grid, in the same string format.
    for name, t in (("zeroDopplerStartTime", win.t_first), ("zeroDopplerEndTime", win.t_last)):
        if name in ident:
            ds = ident[name]
            up.datasets[ds.name] = _string_like(ds, _format_time(t, prod.epoch, read_string(ds)))

    # Bounding polygon and frame coverage.
    if geo is not None and "boundingPolygon" in ident:
        import shapely
        from shapely.geometry.polygon import orient

        bp = ident["boundingPolygon"]
        poly = _window_polygon(geo, win, up.h_ref)
        try:
            old_poly = shapely.from_wkt(read_string(bp))
        except Exception:
            old_poly = None
        epsg = int(np.asarray(bp.attrs.get("epsg", geo.epsg)).ravel()[0])
        if epsg != geo.epsg:
            from pyproj import Transformer
            tr = Transformer.from_crs(geo.epsg, epsg, always_xy=True)
            poly = shapely.transform(poly, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
        if old_poly is not None and old_poly.geom_type == "Polygon":
            poly = orient(poly, sign=1.0 if old_poly.exterior.is_ccw else -1.0)
        up.wkt = shapely.to_wkt(poly, rounding_precision=8, trim=True)
        up.datasets[bp.name] = _string_like(bp, up.wkt)

        if "isFullFrame" in ident:
            ff = ident["isFullFrame"]
            up.datasets[ff.name] = _string_like(ff, "False")
            if old_poly is not None and old_poly.area > 0 and "frameCoveragePercentage" in ff.attrs:
                old = np.asarray(ff.attrs["frameCoveragePercentage"])
                ratio = min(1.0, poly.area / old_poly.area)
                up.attrs[ff.name] = {"frameCoveragePercentage": (old * ratio).astype(old.dtype)}

    # Lists of frequencies / polarizations if anything was dropped.
    if "listOfFrequencies" in ident and freqs != prod.frequencies:
        ds = ident["listOfFrequencies"]
        up.datasets[ds.name] = _string_like(ds, freqs)
    for fr in freqs:
        g = prod.swaths[f"frequency{fr}"]
        if "listOfPolarizations" in g and pols[fr] != prod.pols[fr]:
            ds = g["listOfPolarizations"]
            up.datasets[ds.name] = _string_like(ds, pols[fr])
    return up


def _finalize(prod: RslcProduct, dst: h5py.File, win: RadarWindow, freqs: list[str],
              stats: dict, recompute_stats: bool, up: _Updates, source_label: str) -> None:
    swaths = dst[prod.swaths.name]
    geo = prod.geolocation_grid()

    for path, attrs in up.attrs.items():
        for k, v in attrs.items():
            dst[path].attrs.create(k, data=v)

    # Scene-centre spacings, evaluated at the centre of the new grid.
    if geo is not None:
        t_full = 0.5 * (prod.t[0] + prod.t[-1])
        t_new = 0.5 * (win.t_first + win.t_last)
        for fr in freqs:
            g = swaths[f"frequency{fr}"]
            r_full = 0.5 * (prod.ranges[fr][0] + prod.ranges[fr][-1])
            a, b = win.cols[fr]
            r_new = 0.5 * (prod.ranges[fr][a] + prod.ranges[fr][b - 1])
            try:
                az0, rg0 = _ground_spacing(geo, t_full, r_full, up.h_ref, prod.dt, prod.dr[fr])
                az1, rg1 = _ground_spacing(geo, t_new, r_new, up.h_ref, prod.dt, prod.dr[fr])
            except Exception as exc:  # pragma: no cover
                log.warning("Could not re-evaluate scene-centre spacing: %s", exc)
                continue
            for name, ratio in (("sceneCenterAlongTrackSpacing", az1 / az0),
                                ("sceneCenterGroundRangeSpacing", rg1 / rg0)):
                if name in g and np.isfinite(ratio) and ratio > 0:
                    g[name][()] = g[name][()] * ratio

    # Image statistics attributes.
    if recompute_stats:
        for path, comp in stats.items():
            ds = dst[path]
            for attr in list(ds.attrs):
                kind = _stat_kind(attr)
                if kind is None:
                    continue
                c, k = kind
                old_dtype = ds.attrs.get_id(attr).dtype
                ds.attrs.create(attr, data=np.asarray(comp[c].value(k), dtype=old_dtype))

    # Provenance (CF "history" attribute on the root group).
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    note = (f"{stamp}: spatial subset of {Path(str(source_label)).name} by nisar-subset "
            f"{__version__}; {win}")
    prev = dst.attrs.get("history")
    if prev is not None:
        prev = prev.decode() if isinstance(prev, bytes) else str(prev)
        note = prev + "\n" + note
    dst.attrs["history"] = np.bytes_(note)



def default_output_name(label: str) -> str:
    base = Path(str(label).split("?")[0]).name
    stem = base[:-3] if base.endswith(".h5") else base
    return f"{stem}_subset.h5"


def write_subset(prod: RslcProduct, win: RadarWindow, output, *,
                 frequencies: Sequence[str] | None = None,
                 polarizations: Sequence[str] | None = None,
                 crop_metadata: bool = True, lut_margin: int = 3,
                 recompute_stats: bool = True, overwrite: bool = False,
                 source_label: str = "") -> SubsetResult:
    """Write the subset product for an already computed window."""
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists (use overwrite=True)")

    freqs = [f for f in (frequencies or prod.frequencies)]
    bad = set(freqs) - set(prod.frequencies)
    if bad:
        raise ValueError(f"Frequencies {sorted(bad)} not in product ({prod.frequencies})")
    if polarizations:
        wanted = [p.upper() for p in polarizations]
        pols = {fr: [p for p in prod.pols[fr] if p in wanted] for fr in freqs}
        missing = set(wanted) - set().union(*map(set, pols.values()))
        if missing:
            raise ValueError(f"Polarizations {sorted(missing)} not in product ({prod.pols})")
        empty = [fr for fr in freqs if not pols[fr]]
        if empty:
            if frequencies:
                raise ValueError(f"No selected polarization exists for frequency {empty}")
            freqs = [fr for fr in freqs if pols[fr]]
            pols = {fr: pols[fr] for fr in freqs}
    else:
        pols = {fr: list(prod.pols[fr]) for fr in freqs}
    win.cols = {fr: c for fr, c in win.cols.items() if fr in freqs}

    planner = _Planner(prod, win, freqs, pols, crop_metadata, lut_margin)
    updates = _plan_updates(prod, win, freqs, pols)
    src = prod.file
    tmp = output.with_name(output.name + ".part")
    output.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()

    # Same file creation properties as the source (userblock, file-space
    # strategy / paged aggregation, B-tree sizes...).
    fcpl = src.id.get_create_plist()
    fid = h5py.h5f.create(os.fsencode(tmp), h5py.h5f.ACC_TRUNC, fcpl=fcpl)
    try:
        with h5py.File(fid) as dst:
            copy_attrs(src["/"], dst["/"])
            stats: dict = {}
            _copy_tree(src["/"], dst["/"], planner, stats, updates.datasets)
            n = copy_dimension_scales(src, dst)
            log.info("Re-attached %d dimension scales", n)
            _finalize(prod, dst, win, freqs, stats, recompute_stats, updates,
                      source_label or src.filename)
        os.replace(tmp, output)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise

    shapes = {fr: (win.length, win.width(fr)) for fr in freqs}
    return SubsetResult(output, win, freqs, pols, updates.wkt, shapes)


def subset_rslc(source, aoi=None, output=None, *, aoi_epsg: int = 4326,
                rows: Sequence[int] | None = None, cols: Sequence[int] | None = None,
                frequencies: Sequence[str] | None = None,
                polarizations: Sequence[str] | None = None,
                pad: int = 0, heights: Iterable[float] | None = None,
                crop_metadata: bool = True, lut_margin: int = 3,
                recompute_stats: bool = True, overwrite: bool = False,
                mode: str = "stream", access: str = "external",
                download_dir=None, short_names=None) -> SubsetResult:
    """Subset a NISAR RSLC to an AOI and save it as a NISAR-format RSLC.

    Parameters
    ----------
    source
        Local RSLC path, an open ``h5py.File``, or a NISAR granule ID such as
        ``NISAR_L1_PR_RSLC_004_122_D_067_4005_DHDH_A_20251106T160541_...``
        (found in CMR and streamed or downloaded with earthaccess).
    aoi
        Area of interest: WKT, GeoJSON (str/dict/file), shapely geometry or a
        ``(min_x, min_y, max_x, max_y)`` box, in ``aoi_epsg`` (default 4326).
    output
        Output path; defaults to ``<input name>_subset.h5`` in the current
        directory.
    rows, cols
        Explicit ``(start, stop)`` index window instead of an AOI. ``cols``
        refer to the first selected frequency.
    frequencies, polarizations
        Keep only these (default: all).
    pad
        Extra lines/samples added on every side of the window.
    heights
        Terrain heights (m above ellipsoid) used to map the AOI into radar
        coordinates. Default: the product's mean ``referenceTerrainHeight``
        over the AOI. In steep terrain pass the expected min and max height to
        guarantee coverage, e.g. ``heights=(0, 2500)``.
    crop_metadata, lut_margin
        Crop metadata LUTs to the window plus ``lut_margin`` LUT nodes
        (default) or keep them in full.
    recompute_stats
        Recompute the image statistics attributes for the subset.
    mode, access, download_dir, short_names
        Remote options for granule IDs: ``mode="stream"`` (default) or
        ``"download"``; ``access="external"`` (HTTPS) or ``"direct"`` (S3,
        only from AWS us-west-2).
    """
    remote_kw = dict(mode=mode, access=access, download_dir=download_dir, short_names=short_names)
    with open_source(source, **remote_kw) as (f, label):
        prod = RslcProduct(f)
        freqs = list(frequencies or prod.frequencies)
        win = compute_window(prod, aoi, aoi_epsg=aoi_epsg, rows=rows, cols=cols,
                             frequencies=freqs, pad=pad, heights=heights)
        log.info("Radar window: %s", win)
        out = Path(output) if output else Path(default_output_name(label))
        return write_subset(prod, win, out, frequencies=frequencies,
                            polarizations=polarizations, crop_metadata=crop_metadata,
                            lut_margin=lut_margin, recompute_stats=recompute_stats,
                            overwrite=overwrite, source_label=label)
