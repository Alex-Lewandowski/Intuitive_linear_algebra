"""Tests for nisar_subset (no ISCE3 needed).

The fixture is a real-geometry RSLC (see make_fixture.py). Expected radar
extents of the AOIs below were computed with ISCE3 geo2rdr (exact, h = 0 m).
"""
import json
import subprocess
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
import shapely

from nisar_subset import RslcProduct, load_aoi, subset_rslc

FIXTURE = Path(__file__).parent / "data" / "envisat_rslc_fixture.h5"
P = "/science/LSAR"
S = f"{P}/SLC"

AOI = "POLYGON ((-115.62 34.826, -115.58 34.83, -115.59 34.835, -115.61 34.832, -115.62 34.826))"
# ISCE3 geo2rdr extent of AOI (fractional line / sample indices).
AOI_EXACT = dict(rows=(56.88, 439.79), cols=(277.03, 424.56))


@pytest.fixture(scope="module")
def subset(tmp_path_factory):
    out = tmp_path_factory.mktemp("s") / "sub.h5"
    res = subset_rslc(FIXTURE, AOI, out)
    return res


def _datasets(f):
    out = {}
    f.visititems(lambda n, o: out.__setitem__(n, o) if isinstance(o, h5py.Dataset) else None)
    return out


def _groups(f):
    out = []
    f.visititems(lambda n, o: out.append(n) if isinstance(o, h5py.Group) else None)
    return out


# --- window ------------------------------------------------------------------
def test_window_is_tight_and_covers_aoi(subset):
    w = subset.window
    lo, hi = AOI_EXACT["rows"]
    assert w.row_start <= np.floor(lo) and w.row_stop - 1 >= np.ceil(hi)
    assert w.row_start >= np.floor(lo) - 1 and w.row_stop - 1 <= np.ceil(hi) + 2
    lo, hi = AOI_EXACT["cols"]
    c0, c1 = w.cols["A"]
    assert c0 <= np.floor(lo) and c1 - 1 >= np.ceil(hi)
    assert c0 >= np.floor(lo) - 1 and c1 - 1 <= np.ceil(hi) + 2


def test_point_aoi_gives_small_window(tmp_path):
    res = subset_rslc(FIXTURE, "POINT (-115.58 34.83)", tmp_path / "p.h5")
    assert res.window.length <= 3 and res.window.width("A") <= 3


def test_heights_widen_window(tmp_path):
    flat = subset_rslc(FIXTURE, AOI, tmp_path / "a.h5").window
    hilly = subset_rslc(FIXTURE, AOI, tmp_path / "b.h5", heights=(0, 2000)).window
    assert hilly.cols["A"][0] < flat.cols["A"][0] or hilly.cols["A"][1] > flat.cols["A"][1]


def test_aoi_outside_scene_raises(tmp_path):
    with pytest.raises(ValueError, match="does not intersect"):
        subset_rslc(FIXTURE, "POLYGON ((10 10, 11 10, 11 11, 10 10))", tmp_path / "x.h5")
    assert not (tmp_path / "x.h5").exists()


def test_partial_overlap_warns_and_clips(tmp_path):
    with pytest.warns(UserWarning, match="outside"):
        res = subset_rslc(FIXTURE, (-115.70, 34.80, -115.55, 34.83), tmp_path / "c.h5")
    assert res.window.row_start == 0 or res.window.cols["A"][1] == 500


# --- layout ------------------------------------------------------------------
def test_same_layout_as_input(subset):
    with h5py.File(FIXTURE) as a, h5py.File(subset.output) as b:
        assert _groups(a) == _groups(b)
        da, db = _datasets(a), _datasets(b)
        assert list(da) == list(db)  # same names, same (creation) order
        for name in da:
            x, y = da[name], db[name]
            if x.dtype.kind == "S":  # rewritten strings may need more room ("True" -> "False")
                assert y.dtype.kind == "S" and y.dtype.itemsize >= x.dtype.itemsize, name
            else:
                assert x.dtype == y.dtype, name
            assert x.ndim == y.ndim, name
            assert x.compression == y.compression and x.shuffle == y.shuffle, name
            assert (x.chunks is None) == (y.chunks is None), name
            assert set(x.attrs) == set(y.attrs), name
        assert set(a.attrs) | {"history"} == set(b.attrs)


def test_image_is_exact_window(subset):
    w = subset.window
    c0, c1 = w.cols["A"]
    with h5py.File(FIXTURE) as a, h5py.File(subset.output) as b:
        for pol in ("HH", "HV"):
            src = a[f"{S}/swaths/frequencyA/{pol}"][w.row_start:w.row_stop, c0:c1]
            dst = b[f"{S}/swaths/frequencyA/{pol}"]
            assert dst.dtype.names == ("r", "i")  # NISAR complex32 kept
            np.testing.assert_array_equal(dst[()], src)
            assert dst.chunks == (128, 128)
        np.testing.assert_array_equal(b[f"{S}/swaths/zeroDopplerTime"][()],
                                      a[f"{S}/swaths/zeroDopplerTime"][w.row_start:w.row_stop])
        np.testing.assert_array_equal(b[f"{S}/swaths/frequencyA/slantRange"][()],
                                      a[f"{S}/swaths/frequencyA/slantRange"][c0:c1])
        for k in ("zeroDopplerTimeSpacing", "frequencyA/slantRangeSpacing",
                  "frequencyA/processedCenterFrequency", "frequencyA/numberOfSubSwaths"):
            assert a[f"{S}/swaths/{k}"][()] == b[f"{S}/swaths/{k}"][()]
        assert b[f"{S}/swaths/zeroDopplerTime"].attrs["units"] == \
            a[f"{S}/swaths/zeroDopplerTime"].attrs["units"]


def test_valid_samples_shifted(subset):
    w = subset.window
    c0, c1 = w.cols["A"]
    with h5py.File(FIXTURE) as a, h5py.File(subset.output) as b:
        for k in (1, 2):
            src = a[f"{S}/swaths/frequencyA/validSamplesSubSwath{k}"][w.row_start:w.row_stop].astype(int)
            dst = b[f"{S}/swaths/frequencyA/validSamplesSubSwath{k}"][()]
            assert dst.dtype == np.uint32
            exp = np.clip(src - c0, 0, c1 - c0)
            exp[exp[:, 0] >= exp[:, 1]] = 0
            np.testing.assert_array_equal(dst, exp)


def test_statistics_recomputed(subset):
    with h5py.File(subset.output) as b:
        ds = b[f"{S}/swaths/frequencyA/HH"]
        data = ds[()]
        re_, im_ = data["r"].astype(np.float64), data["i"].astype(np.float64)
        np.testing.assert_allclose(ds.attrs["mean_real_value"], re_.mean(), rtol=1e-5)
        np.testing.assert_allclose(ds.attrs["max_imag_value"], im_.max(), rtol=1e-6)
        np.testing.assert_allclose(ds.attrs["sample_stddev_real"], re_.std(ddof=1), rtol=1e-5)
        assert ds.attrs["mean_real_value"].dtype == np.float32


def test_dimension_scales_reattached(subset):
    with h5py.File(subset.output) as b:
        g = b[f"{S}/metadata/geolocationGrid"]
        for name in ("coordinateX", "incidenceAngle"):
            dims = [list(g[name].dims[i].values()) for i in range(3)]
            assert [d[0].name.rsplit("/", 1)[-1] for d in dims] == \
                ["heightAboveEllipsoid", "zeroDopplerTime", "slantRange"]
            assert g[name].shape[1:] == (g["zeroDopplerTime"].size, g["slantRange"].size)
        assert h5py.h5ds.is_scale(g["zeroDopplerTime"].id)


def test_luts_cropped_consistently_and_cover_grid(subset):
    with h5py.File(FIXTURE) as a, h5py.File(subset.output) as b:
        t = b[f"{S}/swaths/zeroDopplerTime"][()]
        r = b[f"{S}/swaths/frequencyA/slantRange"][()]
        # (LUT group, group holding its axes): the fixture mixes the current
        # layout (axes next to the LUT) and the legacy one (axes in a parent).
        cases = [("metadata/geolocationGrid",) * 2,
                 ("metadata/calibrationInformation/geometry", "metadata/calibrationInformation"),
                 ("metadata/calibrationInformation/frequencyA/noiseEquivalentBackscatter",) * 2,
                 ("metadata/processingInformation/parameters",) * 2,
                 ("metadata/processingInformation/parameters/frequencyA",
                  "metadata/processingInformation/parameters")]
        for grp, axes in cases:
            gb, ga = b[f"{S}/{grp}"], a[f"{S}/{grp}"]
            tb, rb = b[f"{S}/{axes}/zeroDopplerTime"][()], b[f"{S}/{axes}/slantRange"][()]
            ta, ra = a[f"{S}/{axes}/zeroDopplerTime"][()], a[f"{S}/{axes}/slantRange"][()]
            assert tb[0] <= t[0] and tb[-1] >= t[-1], grp
            assert rb[0] <= r[0] and rb[-1] >= r[-1], grp
            assert tb.size < ta.size and rb.size < ra.size, grp
            i0, j0 = int(np.searchsorted(ta, tb[0])), int(np.searchsorted(ra, rb[0]))
            n2d = 0
            for name, ds in gb.items():
                if isinstance(ds, h5py.Dataset) and ds.ndim >= 2:
                    exp = ga[name][..., i0:i0 + tb.size, j0:j0 + rb.size]
                    np.testing.assert_array_equal(ds[()], exp)
                    n2d += 1
            assert n2d, grp
        # Doppler on the shared parameters axes.
        dop = b[f"{S}/metadata/processingInformation/parameters/frequencyA/dopplerCentroid"]
        pr = b[f"{S}/metadata/processingInformation/parameters"]
        assert dop.shape == (pr["zeroDopplerTime"].size, pr["slantRange"].size)
        # Not functions of the LUT axes: untouched.
        assert b[f"{S}/metadata/processingInformation/parameters/rangeChirpWeighting"].shape == (256,)
        # Orbit kept in full.
        assert b[f"{S}/metadata/orbit/time"].shape == a[f"{S}/metadata/orbit/time"].shape


def test_no_crop_metadata(tmp_path):
    res = subset_rslc(FIXTURE, AOI, tmp_path / "n.h5", crop_metadata=False)
    with h5py.File(FIXTURE) as a, h5py.File(res.output) as b:
        for k in ("geolocationGrid/coordinateX", "calibrationInformation/geometry/beta0"):
            assert a[f"{S}/metadata/{k}"].shape == b[f"{S}/metadata/{k}"].shape


# --- identification ----------------------------------------------------------
def test_identification_updated(subset):
    w = subset.window
    with h5py.File(FIXTURE) as a, h5py.File(subset.output) as b:
        ident = b[f"{P}/identification"]
        t = a[f"{S}/swaths/zeroDopplerTime"]
        units = t.attrs["units"]
        units = units.decode() if isinstance(units, bytes) else units
        epoch = np.datetime64(units.split("since ")[1].strip().replace(" ", "T"), "ns")
        start = epoch + np.timedelta64(int(round(t[w.row_start] * 1e9)), "ns")
        end = epoch + np.timedelta64(int(round(t[w.row_stop - 1] * 1e9)), "ns")
        assert np.datetime64(ident["zeroDopplerStartTime"][()].decode()) == start
        assert np.datetime64(ident["zeroDopplerEndTime"][()].decode()) == end
        assert len(ident["zeroDopplerStartTime"][()]) == len(a[f"{P}/identification/zeroDopplerStartTime"][()])

        poly = shapely.from_wkt(ident["boundingPolygon"][()].decode())
        full = shapely.from_wkt(a[f"{P}/identification/boundingPolygon"][()].decode())
        aoi = shapely.from_wkt(AOI)
        assert poly.is_valid and poly.area < full.area
        assert poly.buffer(1e-4).contains(aoi)            # covers the AOI
        assert poly.area < 2.0 * aoi.envelope.area        # and is not much bigger
        assert ident["boundingPolygon"].attrs["epsg"] == a[f"{P}/identification/boundingPolygon"].attrs["epsg"]

        assert ident["isFullFrame"][()] == b"False"
        cov = ident["isFullFrame"].attrs["frameCoveragePercentage"]
        assert 0 < cov < 100
        assert "spatial subset" in b.attrs["history"].decode()


def test_scene_center_spacing_updated(subset):
    with h5py.File(FIXTURE) as a, h5py.File(subset.output) as b:
        k = f"{S}/swaths/frequencyA/sceneCenterGroundRangeSpacing"
        old, new = a[k][()], b[k][()]
        assert new != old and abs(new / old - 1) < 0.05   # far-range subset: slightly finer
        k = f"{S}/swaths/frequencyA/sceneCenterAlongTrackSpacing"
        assert abs(b[k][()] / a[k][()] - 1) < 0.01


# --- selections --------------------------------------------------------------
def test_polarization_selection(tmp_path):
    res = subset_rslc(FIXTURE, AOI, tmp_path / "hv.h5", polarizations=["hv"])
    with h5py.File(res.output) as b:
        sw = b[f"{S}/swaths/frequencyA"]
        assert "HH" not in sw and "HV" in sw
        assert list(sw["listOfPolarizations"][()]) == [b"HV"]
        assert sw["listOfPolarizations"].dtype == np.dtype("S2")
        assert "HH" not in b[f"{S}/metadata/calibrationInformation/frequencyA/noiseEquivalentBackscatter"]
        assert "HH" not in b[f"{S}/metadata/calibrationInformation/frequencyA"]
    with pytest.raises(ValueError, match="Polarizations"):
        subset_rslc(FIXTURE, AOI, tmp_path / "vv.h5", polarizations=["VV"])


def test_rows_cols_window(tmp_path):
    res = subset_rslc(FIXTURE, rows=(10, 60), cols=(100, 180), output=tmp_path / "rc.h5", pad=2)
    assert (res.window.row_start, res.window.row_stop) == (8, 62)
    assert res.window.cols["A"] == (98, 182)
    with h5py.File(res.output) as b:
        assert b[f"{S}/swaths/frequencyA/HH"].shape == (54, 84)


def test_overwrite_protection(tmp_path, subset):
    out = tmp_path / "o.h5"
    subset_rslc(FIXTURE, AOI, out)
    with pytest.raises(FileExistsError):
        subset_rslc(FIXTURE, AOI, out)
    subset_rslc(FIXTURE, AOI, out, overwrite=True)


def test_open_file_object_input(tmp_path):
    with open(FIXTURE, "rb") as fo, h5py.File(fo, "r") as f:
        res = subset_rslc(f, AOI, tmp_path / "fo.h5")
    ref = subset_rslc(FIXTURE, AOI, tmp_path / "ref.h5")
    assert str(res.window) == str(ref.window)


def test_product_view():
    with h5py.File(FIXTURE) as f:
        p = RslcProduct(f)
        assert p.frequencies == ["A"] and p.pols == {"A": ["HH", "HV"]}
        assert p.shape == {"A": (500, 500)}


# --- AOI parsing -------------------------------------------------------------
def test_load_aoi_variants(tmp_path):
    poly = shapely.from_wkt(AOI)
    gj = json.dumps({"type": "Feature", "properties": {}, "geometry": shapely.geometry.mapping(poly)})
    (tmp_path / "a.geojson").write_text(gj)
    (tmp_path / "a.wkt").write_text(AOI)
    for v in (AOI, gj, json.loads(gj), str(tmp_path / "a.geojson"), tmp_path / "a.wkt", poly):
        assert load_aoi(v).equals(poly)
    assert load_aoi((0, 1, 2, 3)).bounds == (0, 1, 2, 3)


def test_aoi_in_other_crs(tmp_path):
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, 32611, always_xy=True)
    utm = shapely.transform(shapely.from_wkt(AOI), lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
    a = subset_rslc(FIXTURE, utm, tmp_path / "u.h5", aoi_epsg=32611).window
    b = subset_rslc(FIXTURE, AOI, tmp_path / "g.h5").window
    assert abs(a.row_start - b.row_start) <= 1 and abs(a.cols["A"][1] - b.cols["A"][1]) <= 1


# --- CLI ---------------------------------------------------------------------
def test_cli(tmp_path):
    out = tmp_path / "cli.h5"
    r = subprocess.run([sys.executable, "-m", "nisar_subset", str(FIXTURE), "--aoi", AOI,
                        "-o", str(out), "--pols", "HH"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "Wrote" in r.stdout and out.exists()
    r = subprocess.run([sys.executable, "-m", "nisar_subset", str(FIXTURE), "-o", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 2


# --- granule IDs (earthaccess replaced by a local stand-in) -------------------
class _FakeGranule(dict):
    def __init__(self, path):
        super().__init__(umm={"GranuleUR": Path(path).stem})
        self.path = str(path)

    def data_links(self, access="external"):
        return [self.path]


@pytest.fixture
def fake_earthaccess(monkeypatch):
    import earthaccess
    import fsspec

    calls = {"login": 0, "search": []}

    def search_data(count=-1, **kw):
        calls["search"].append(kw)
        return [_FakeGranule(FIXTURE)] if kw.get("short_name") == "NISAR_L1_RSLC_BETA_V1" else []

    def download(granules, local_path=".", **kw):
        import shutil
        dst = Path(local_path) / Path(granules[0].path).name
        shutil.copy(granules[0].path, dst)
        return [dst]

    monkeypatch.setattr(earthaccess, "login", lambda **kw: calls.__setitem__("login", calls["login"] + 1))
    monkeypatch.setattr(earthaccess, "search_data", search_data)
    monkeypatch.setattr(earthaccess, "download", download)
    monkeypatch.setattr(earthaccess, "get_fsspec_https_session", lambda: fsspec.filesystem("file"))
    return calls


def test_granule_id_streaming(tmp_path, fake_earthaccess):
    gid = "NISAR_L1_PR_RSLC_004_122_D_067_4005_DHDH_A_20251106T160541_20251106T160622_X05009_N_F_J_001"
    res = subset_rslc(gid, AOI, tmp_path / "remote.h5")
    ref = subset_rslc(FIXTURE, AOI, tmp_path / "local.h5")
    assert fake_earthaccess["login"] == 1
    assert fake_earthaccess["search"][0] == {"short_name": "NISAR_L1_RSLC_V1", "granule_name": gid}
    assert str(res.window) == str(ref.window)
    with h5py.File(res.output) as a, h5py.File(ref.output) as b:
        np.testing.assert_array_equal(a[f"{S}/swaths/frequencyA/HH"][()], b[f"{S}/swaths/frequencyA/HH"][()])


def test_granule_id_download(tmp_path, fake_earthaccess):
    res = subset_rslc("NISAR_L1_PR_RSLC_TEST.h5", AOI, tmp_path / "dl_sub.h5", mode="download",
                      download_dir=tmp_path / "dl")
    assert (tmp_path / "dl" / FIXTURE.name).exists() and res.output.exists()


def test_default_output_name(tmp_path, monkeypatch, fake_earthaccess):
    monkeypatch.chdir(tmp_path)
    res = subset_rslc(str(FIXTURE), AOI)
    assert res.output == Path("envisat_rslc_fixture_subset.h5")


def test_geolocation_grid_across_antimeridian(tmp_path):
    from nisar_subset.geometry import GeolocationGrid, aoi_radar_extent

    t = np.linspace(0, 10, 11)
    r = np.linspace(800e3, 900e3, 11)
    tt, rr = np.meshgrid(t, r, indexing="ij")
    lon = 179.8 + 0.04 * tt + 0.004 * (rr - 800e3) / 1e3     # 179.8 .. 180.6
    lon = np.where(lon > 180, lon - 360, lon)
    lat = 52.0 + 0.01 * tt
    with h5py.File(tmp_path / "g.h5", "w") as f:
        g = f.create_group("geo")
        g["coordinateX"] = np.stack([lon, lon])
        g["coordinateY"] = np.stack([lat, lat])
        g["heightAboveEllipsoid"] = np.array([0.0, 100.0])
        g["zeroDopplerTime"] = t
        g["zeroDopplerTime"].attrs["units"] = np.bytes_("seconds since 2026-01-01T00:00:00")
        g["slantRange"] = r
        g["epsg"] = np.int32(4326)
        grid = GeolocationGrid(g, np.datetime64("2026-01-01T00:00:00", "ns"))
        assert grid.wrap
        # A box straddling the antimeridian at t = 4..6.
        aoi = shapely.MultiPolygon([shapely.box(179.99, 52.042, 180.0, 52.046),
                                    shapely.box(-180.0, 52.042, -179.99, 52.046)])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # fully inside: no clipping warning
            t0, t1, r0, r1 = aoi_radar_extent(grid, aoi, [0.0])
        assert 4.0 < t0 < t1 < 4.8
        x, y = grid.forward(np.array([5.0]), np.array([850e3]), 0.0)
        assert -180 <= x[0] <= 180
