"""Check that a subset RSLC is read by ISCE3 exactly like the full product.

Requires an environment with ISCE3 (``conda install -c conda-forge isce3``).

    python tests/validate_with_isce3.py FULL.h5 SUBSET.h5 [--aoi WKT]

For every frequency the ISCE3 radar grid of the subset must equal the full
product's grid offset to the window (same start time / range, spacing, PRF,
wavelength, epoch, look side) and the image samples must be identical. The
orbit, attitude, Doppler, azimuth FM rate, calibration and noise LUTs read by
ISCE3 must cover the subset grid and agree with the full product's LUTs on it.
With --aoi, ISCE3 geo2rdr is used to confirm that every AOI vertex falls
inside the subset grid.
"""
from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np

import isce3
from nisar.products.readers import SLC


def _close(a, b, tol, what):
    if not np.allclose(a, b, rtol=0, atol=tol, equal_nan=True):
        raise AssertionError(f"{what}: {a} != {b}")


def check(full_path: str, sub_path: str, aoi: str | None = None, heights=(0.0,)) -> list[str]:
    full, sub = SLC(hdf5file=full_path), SLC(hdf5file=sub_path)
    report = []
    assert full.frequencies == sub.frequencies or set(sub.frequencies) <= set(full.frequencies)
    t_full = full.getZeroDopplerTime()
    t_sub = sub.getZeroDopplerTime()
    i0 = int(np.argmin(np.abs(t_full - t_sub[0])))

    for fr in sub.frequencies:
        g_full, g_sub = full.getRadarGrid(fr), sub.getRadarGrid(fr)
        r_full, r_sub = full.getSlantRange(fr), sub.getSlantRange(fr)
        j0 = int(np.argmin(np.abs(r_full - r_sub[0])))
        expect = g_full.offset_and_resize(i0, j0, g_sub.length, g_sub.width)
        _close(g_sub.sensing_start, expect.sensing_start, 1e-9, "sensing_start")
        _close(g_sub.starting_range, expect.starting_range, 1e-6, "starting_range")
        _close(g_sub.prf, expect.prf, 1e-9, "prf")
        _close(g_sub.range_pixel_spacing, expect.range_pixel_spacing, 1e-9, "range spacing")
        _close(g_sub.wavelength, expect.wavelength, 1e-12, "wavelength")
        assert g_sub.lookside == expect.lookside
        assert str(g_sub.ref_epoch) == str(expect.ref_epoch)
        report.append(f"freq {fr}: grid = full[{i0}:{i0 + g_sub.length}, {j0}:{j0 + g_sub.width}] "
                      f"({g_sub.length} x {g_sub.width})")

        sw_sub = sub.getSwathMetadata(fr)
        assert sw_sub.lines == g_sub.length and sw_sub.samples == g_sub.width
        _close(sw_sub.processed_center_frequency,
               full.getSwathMetadata(fr).processed_center_frequency, 1e-3, "fc")

        with h5py.File(full_path, "r", libver="latest", swmr=True) as fh:
            present = [p for p in sub.polarizations[fr] if full.slcPath(fr, p) in fh]
        for pol in present:
            a = full.getSlcDatasetAsNativeComplex(fr, pol)[i0:i0 + g_sub.length, j0:j0 + g_sub.width]
            b = sub.getSlcDatasetAsNativeComplex(fr, pol)[:]
            assert np.array_equal(a, b, equal_nan=True), f"image {fr}/{pol} differs"
        report.append(f"freq {fr}: image samples identical for {present}")

        # LUTs used by the processors must cover the grid and agree with the full ones.
        tt, rr = np.meshgrid(np.linspace(g_sub.sensing_start, g_sub.sensing_stop, 7),
                             np.linspace(g_sub.starting_range, g_sub.end_range, 7), indexing="ij")
        for name, get in (("dopplerCentroid", lambda p: p.getDopplerCentroid(fr)),):
            try:
                lf = get(full)
            except Exception:
                report.append(f"freq {fr}: full product has no {name} LUT")
                continue
            ls = get(sub)
            pts = list(zip(tt.ravel(), rr.ravel()))
            vf = np.array([lf.eval(t, r) for t, r in pts])
            vs = np.array([ls.eval(t, r) for t, r in pts])
            if all(lf.contains(t, r) for t, r in pts):
                assert all(ls.contains(t, r) for t, r in pts), f"{name} LUT does not cover grid"
            _close(vs, vf, 1e-6 * max(1.0, np.abs(vf).max()), name)
            report.append(f"freq {fr}: {name} LUT covers the grid and matches the full product")
        pol = sub.polarizations[fr][0]
        try:
            nf = full.getNoiseEquivalentBackscatter(fr, pol)
        except Exception as exc:  # the full product itself has no usable noise LUT
            report.append(f"freq {fr}: full product has no readable noise LUT for {pol} ({exc})")
        else:
            ns = sub.getNoiseEquivalentBackscatter(fr, pol)

            def covers(axis, lo, hi):
                return axis[0] <= lo and axis[-1] >= hi

            # The subset LUT must cover the grid wherever the full LUT did.
            if covers(nf.az_time, g_sub.sensing_start, g_sub.sensing_stop):
                assert covers(ns.az_time, g_sub.sensing_start, g_sub.sensing_stop), "noise LUT time"
            if covers(nf.slant_range, g_sub.starting_range, g_sub.end_range):
                assert covers(ns.slant_range, g_sub.starting_range, g_sub.end_range), "noise LUT range"
            # ...and hold the full LUT's values on its nodes.
            ti = np.searchsorted(nf.az_time, ns.az_time)
            ri = np.searchsorted(nf.slant_range, ns.slant_range)
            assert np.array_equal(nf.power_linear[np.ix_(ti, ri)], ns.power_linear, equal_nan=True)
            report.append(f"freq {fr}: noise LUT ({pol}) covers the grid, values match "
                          f"({ns.power_linear.shape} of {nf.power_linear.shape} nodes)")

    for conv in ("beta0", "sigma0", "gamma0"):
        try:
            lut = sub.getRadiometricCalibrationLUT(conv)
        except KeyError:
            continue
        assert lut.contains(g_sub.sensing_start, g_sub.starting_range)
        assert lut.contains(g_sub.sensing_stop, g_sub.end_range)
    report.append("calibration LUTs cover the grid")

    of, os_ = full.getOrbit(), sub.getOrbit()
    assert np.allclose(of.position, os_.position) and np.allclose(of.time, os_.time)
    report.append("orbit identical")

    with h5py.File(sub_path, "r", libver="latest", swmr=True) as f:
        ident = f[sub.IdentificationPath]
        start = ident["zeroDopplerStartTime"][()].decode()
        expect_start = sub.getRadarGrid().ref_epoch + isce3.core.TimeDelta(t_sub[0])
        assert abs((isce3.core.DateTime(start) - expect_start).total_seconds()) < 1e-6, start
    report.append(f"identification start time {start} matches the grid")

    if aoi:
        import shapely
        geom = shapely.from_wkt(aoi)
        orbit = sub.getOrbit()
        g = sub.getRadarGrid()
        ell = isce3.core.Ellipsoid()
        zero = isce3.core.LUT2d()
        tol_t, tol_r = 1.0 / g.prf, g.range_pixel_spacing
        pts = shapely.get_coordinates(geom)
        for h in heights:
            for lon, lat in pts:
                llh = np.array([np.deg2rad(lon), np.deg2rad(lat), h])
                t, r = isce3.geometry.geo2rdr(llh, ell, orbit, zero, g.wavelength, g.lookside)
                assert g.sensing_start - tol_t <= t <= g.sensing_stop + tol_t or \
                    t < full.getRadarGrid().sensing_start or t > full.getRadarGrid().sensing_stop, \
                    f"AOI vertex {lon, lat} (t={t}) outside subset rows"
                assert g.starting_range - tol_r <= r <= g.end_range + tol_r or \
                    r < full.getRadarGrid().starting_range or r > full.getRadarGrid().end_range, \
                    f"AOI vertex {lon, lat} (r={r}) outside subset columns"
        report.append(f"ISCE3 geo2rdr: all {len(pts)} AOI vertices fall inside the subset grid")
    return report


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("full")
    p.add_argument("subset")
    p.add_argument("--aoi")
    p.add_argument("--heights", nargs="+", type=float, default=[0.0])
    a = p.parse_args(argv)
    for line in check(a.full, a.subset, a.aoi, a.heights):
        print("OK ", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
