"""End-to-end check: geocode the full RSLC and the subset with ISCE3.

Both are geocoded (isce3.geocode.geocode_slc, flat DEM, sub-swath masks from
each product) onto the same lat/lon grid; inside the subset's footprint
(away from its edges, where the interpolation kernel would need pixels the
subset does not have) the two outputs must be identical.

    python tests/geocode_compare_isce3.py FULL.h5 SUBSET.h5
"""
import sys
import tempfile

import numpy as np

import isce3
from nisar.products.readers import SLC


def geocode(path, geogrid, dem, window=None):
    """Geocode frequency A / first pol of `path`; optionally only the block
    `window` = (i0, j0, length, width) of it, cut in memory with ISCE3's own
    RadarGridParameters.offset_and_resize (how ISCE3 processes blocks)."""
    p = SLC(hdf5file=path)
    fr = "A"
    grid = p.getRadarGrid(fr)
    pol = p.polarizations[fr][0]
    data = p.getSlcDatasetAsNativeComplex(fr, pol)[:].astype(np.complex64)
    subswaths = p.getSwathMetadata(fr).sub_swaths()
    if window is not None:
        i0, j0, n, m = window
        grid = grid.offset_and_resize(i0, j0, n, m)
        data = np.ascontiguousarray(data[i0:i0 + n, j0:j0 + m])
        subswaths = None
    out = np.full((geogrid.length, geogrid.width), np.nan + 1j * np.nan, dtype=np.complex64)
    mask = np.zeros((geogrid.length, geogrid.width), dtype=np.uint8)
    isce3.geocode.geocode_slc(
        geo_data_blocks=[out], rdr_data_blocks=[data], dem_raster=dem,
        radargrid=grid, geogrid=geogrid, orbit=p.getOrbit(),
        native_doppler=p.getDopplerCentroid(fr), image_grid_doppler=isce3.core.LUT2d(),
        ellipsoid=isce3.core.Ellipsoid(), threshold_geo2rdr=1e-8, num_iter_geo2rdr=50,
        mask_block=mask, sliced_radargrid=grid, subswaths=subswaths,
        flatten=False, reramp=True, invalid_value=np.nan + 1j * np.nan)
    return out, grid, p


def main(full_path, sub_path):
    full = SLC(hdf5file=full_path)
    g = full.getRadarGrid()
    # A geogrid covering the full scene at ~0.0002 deg, built from rdr2geo of the corners.
    orbit, ell = full.getOrbit(), isce3.core.Ellipsoid()
    lons, lats = [], []
    for t in (g.sensing_start, g.sensing_stop):
        for r in (g.starting_range, g.end_range):
            llh = isce3.geometry.rdr2geo(t, r, orbit, g.lookside, 0.0, g.wavelength,
                                         isce3.geometry.DEMInterpolator(0.0))
            lons.append(np.rad2deg(llh[0])); lats.append(np.rad2deg(llh[1]))
    dx = dy = 0.0002
    x0, y0 = min(lons), max(lats)
    w, l = int((max(lons) - x0) / dx) + 1, int((y0 - min(lats)) / dy) + 1
    geogrid = isce3.product.GeoGridParameters(x0, y0, dx, -dy, w, l, 4326)

    tmp = tempfile.mkdtemp()
    dem_path = f"{tmp}/dem.tif"
    dem = isce3.io.Raster(dem_path, 2, 2, 1, 6, "GTiff")  # 6 = GDT_Float32
    dem.set_geotransform([x0 - 0.1, 0.2 + w * dx, 0, y0 + 0.1, 0, -(0.2 + l * dy)])
    dem.set_epsg(4326)
    del dem
    dem = isce3.io.Raster(dem_path)

    a, _, _ = geocode(full_path, geogrid, dem)
    b, gs, _ = geocode(sub_path, geogrid, dem)
    # Reference: the same block cut out of the full product in memory by ISCE3.
    t_full = full.getZeroDopplerTime()
    sub = SLC(hdf5file=sub_path)
    i0 = int(np.argmin(np.abs(t_full - sub.getZeroDopplerTime()[0])))
    j0 = int(np.argmin(np.abs(full.getSlantRange("A") - sub.getSlantRange("A")[0])))
    ref, _, _ = geocode(full_path, geogrid, dem, (i0, j0, gs.length, gs.width))
    # Sub-swath masking is applied to the subset but not the in-memory block.
    valid = np.isfinite(b) & np.isfinite(ref)
    assert valid.sum() > 0
    print(f"subset vs ISCE3 in-memory block of the full product: {int(valid.sum())} pixels, "
          f"max |diff| = {np.abs(b[valid] - ref[valid]).max():.3g}")
    assert np.array_equal(b[valid], ref[valid]), "subset differs from ISCE3's own block of the full product"
    print("OK  bit-identical to geocoding the same block of the full product")

    both = np.isfinite(a) & np.isfinite(b)
    # Interior of the subset footprint: pixels whose interpolation kernel is
    # entirely inside the subset (the sinc kernel is 8 samples wide).
    interior = np.zeros_like(both)
    yy, xx = np.nonzero(both)
    for y, x in zip(yy, xx):
        lon, lat = x0 + (x + 0.5) * dx, y0 - (y + 0.5) * dy
        t, r = isce3.geometry.geo2rdr(np.array([np.deg2rad(lon), np.deg2rad(lat), 0.0]), ell, orbit,
                                      isce3.core.LUT2d(), gs.wavelength, gs.lookside)
        i, j = (t - gs.sensing_start) * gs.prf, (r - gs.starting_range) / gs.range_pixel_spacing
        interior[y, x] = 8 <= i <= gs.length - 9 and 8 <= j <= gs.width - 9
    n = int(interior.sum())
    assert n > 0, "no overlapping interior pixels"
    diff = np.abs(a[interior] - b[interior])
    print(f"geocoded pixels inside subset interior: {n}; max |full - subset| = {diff.max():.3g}; "
          f"max |full| = {np.abs(a[interior]).max():.3g}")
    rel = diff.max() / np.abs(a[interior]).max()
    assert rel < 1e-3, rel
    print(f"OK  full-scene vs subset geocoding agree to {rel:.1e} (relative); the residual is "
          "float rounding from the different block origin, also seen for the in-memory block")


if __name__ == "__main__":
    main(*sys.argv[1:3])
