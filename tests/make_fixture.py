"""Build tests/data/envisat_rslc_fixture.h5 (requires ISCE3; run once).

Starts from ISCE3's ``tests/data/envisat.h5`` RSLC test product (real Envisat
orbit, image and Doppler; Apache-2.0, https://github.com/isce-framework/isce3)
and brings it in line with the current NISAR RSLC layout so the subsetter is
exercised against something realistic:

* ``metadata/geolocationGrid`` is regenerated with ISCE3's own writer
  (``add_geolocation_grid_cubes_to_hdf5``, the call the NISAR RSLC writer
  uses), which also attaches HDF5 dimension scales
* image layers are stored as NISAR ``complex32`` (pairs of float16), chunked
  and gzip-compressed, with the statistics attributes ISCE3 writes
* a second polarization (HV) and ``validSamplesSubSwath1/2`` +
  ``numberOfSubSwaths`` are added; ``isFullFrame`` with its coverage attrs

Usage:  python tests/make_fixture.py /path/to/isce3/tests/data/envisat.h5
"""
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

import isce3
from isce3.core.types import complex32
from nisar.products.readers import SLC
from nisar.workflows.h5_prep import add_geolocation_grid_cubes_to_hdf5

src = Path(sys.argv[1])
out = Path(__file__).parent / "data" / "envisat_rslc_fixture.h5"
out.parent.mkdir(exist_ok=True)
tmp = out.with_suffix(".tmp.h5")
shutil.copy(src, tmp)

slc = SLC(hdf5file=str(tmp))
orbit = slc.getOrbit()
grid = slc.getRadarGrid()
doppler = slc.getDopplerCentroid()
root = "/science/LSAR/SLC"

with h5py.File(tmp, "a") as f:
    sw = f[f"{root}/swaths/frequencyA"]
    # --- geolocation grid as the NISAR RSLC writer makes it ---------------
    del f[f"{root}/metadata/geolocationGrid"]
    heights = np.linspace(-500, 9000, 20)
    # One node every 25 pixels (~110 m x 200 m here; NISAR uses ~500 m on
    # much larger scenes), plus the 5-node margin the NISAR writer adds.
    ysize = max(grid.length // 25, 6)
    xsize = max(grid.width // 25, 6)
    geo_grid = grid.resize_and_keep_startstop(ysize, xsize).add_margin(5, 5)
    add_geolocation_grid_cubes_to_hdf5(f, f"{root}/metadata/geolocationGrid", geo_grid,
                                       heights, orbit, doppler, isce3.core.LUT2d(), 4326)

    # --- complex32 images, chunked + compressed, two polarizations --------
    hh = sw["HH"][()].astype(np.complex64)
    attrs = dict(sw["HH"].attrs)
    del sw["HH"]
    rng = np.random.default_rng(0)
    hv = (0.3 * hh * np.exp(1j * rng.uniform(-np.pi, np.pi, hh.shape))).astype(np.complex64)
    for pol, data in (("HH", hh), ("HV", hv)):
        c32 = np.empty(data.shape, dtype=complex32)
        c32["r"], c32["i"] = data.real, data.imag
        d = sw.create_dataset(pol, data=c32, chunks=(128, 128), compression="gzip",
                              compression_opts=4, shuffle=True)
        for k, v in attrs.items():
            d.attrs[k] = v
        d.attrs["description"] = np.bytes_(f"Focused RSLC image ({pol})")
        re_, im_ = data.real.astype(np.float64), data.imag.astype(np.float64)
        for comp, arr in (("real", re_), ("imag", im_)):
            d.attrs[f"min_{comp}_value"] = np.float32(arr.min())
            d.attrs[f"mean_{comp}_value"] = np.float32(arr.mean())
            d.attrs[f"max_{comp}_value"] = np.float32(arr.max())
            d.attrs[f"sample_stddev_{comp}"] = np.float32(arr.std(ddof=1))
    del sw["listOfPolarizations"]
    lp = sw.create_dataset("listOfPolarizations", data=np.array([b"HH", b"HV"], dtype="S2"))
    lp.attrs["description"] = np.bytes_("List of processed polarization layers with frequency A")

    # --- two sub-swaths with a gap between them ---------------------------
    n_az, n_rg = hh.shape
    s1 = np.tile(np.array([[10, 240]], dtype=np.uint32), (n_az, 1))
    s2 = np.tile(np.array([[260, n_rg - 15]], dtype=np.uint32), (n_az, 1))
    s1[:5] = 0  # a few lines without valid samples
    for i, s in enumerate((s1, s2), 1):
        d = sw.create_dataset(f"validSamplesSubSwath{i}", data=s)
        d.attrs["units"] = np.bytes_("1")
    sw.create_dataset("numberOfSubSwaths", data=np.uint8(2))
    del sw["validSamples"]

    ident = f["/science/LSAR/identification"]
    d = ident.create_dataset("isFullFrame", data=np.bytes_("True"))
    d.attrs["frameCoveragePercentage"] = np.float64(100.0)
    d.attrs["thresholdPercentage"] = np.float64(75.0)

    # Bounding polygon from the new geolocation grid at height 0.
    gg = f[f"{root}/metadata/geolocationGrid"]
    k = int(np.argmin(np.abs(gg["heightAboveEllipsoid"][()])))
    x, y = gg["coordinateX"][k], gg["coordinateY"][k]
    m = 5  # skip the margin nodes
    ring = np.concatenate([np.c_[x[m, m:-m], y[m, m:-m]], np.c_[x[m:-m, -m - 1], y[m:-m, -m - 1]],
                           np.c_[x[-m - 1, m:-m][::-1], y[-m - 1, m:-m][::-1]],
                           np.c_[x[m:-m, m][::-1], y[m:-m, m][::-1]]])
    wkt = "POLYGON ((" + ", ".join(f"{a:.8f} {b:.8f}" for a, b in ring) + \
          f", {ring[0, 0]:.8f} {ring[0, 1]:.8f}))"
    bp = ident["boundingPolygon"]
    battrs = dict(bp.attrs)
    del ident["boundingPolygon"]
    bp = ident.create_dataset("boundingPolygon", data=np.bytes_(wkt))
    for kk, v in battrs.items():
        bp.attrs[kk] = v

# Repack (h5repack keeps dimension scales) to drop space freed by deletions.
subprocess.run([str(Path(sys.executable).with_name("h5repack")), str(tmp), str(out)], check=True)
tmp.unlink()
print("wrote", out, out.stat().st_size // 1024, "KiB")
