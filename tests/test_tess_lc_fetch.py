"""tess_lc_fetch with MAST mocked: de-dupe, cache hits, identical manifest hash, stale fallback."""
import numpy as np
import pytest
from astropy.io import fits

from shared_tools.exo import tess_lc_fetch as f

INV = [
    {"author": "SPOC", "sector": 1, "exptime": 120.0, "productFilename": "tess-s0001-0000000000000007-0120-s_lc.fits",
     "dataURI": "mast:TESS/product/tess-s0001-0000000000000007-0120-s_lc.fits", "size": 1},
    {"author": "SPOC", "sector": 27, "exptime": 20.0, "productFilename": "tess-s0027-0000000000000007-0189-a_fast-lc.fits",
     "dataURI": "mast:TESS/product/tess-s0027-0000000000000007-0189-a_fast-lc.fits", "size": 1},
    {"author": "SPOC", "sector": 27, "exptime": 120.0, "productFilename": "tess-s0027-0000000000000007-0189-s_lc.fits",
     "dataURI": "mast:TESS/product/tess-s0027-0000000000000007-0189-s_lc.fits", "size": 1},
    {"author": "QLP", "sector": 27, "exptime": 600.0, "productFilename": "hlsp_qlp_s0027_v1_llc.fits",
     "dataURI": "mast:HLSP/qlp/s0027/hlsp_qlp_s0027_v1_llc.fits", "size": 1},
    {"author": "QLP", "sector": 27, "exptime": 600.0, "productFilename": "hlsp_qlp_s0027_v2_llc.fits",
     "dataURI": "mast:HLSP/qlp/s0027/hlsp_qlp_s0027_v2_llc.fits", "size": 1},
]


def fake_fits(path):
    t = np.linspace(1000, 1027, 500)
    cols = [fits.Column("TIME", "D", array=t), fits.Column("PDCSAP_FLUX", "E", array=np.ones_like(t)),
            fits.Column("PDCSAP_FLUX_ERR", "E", array=np.full_like(t, 1e-4)),
            fits.Column("QUALITY", "J", array=np.r_[np.zeros(450, int), np.full(50, 32)])]
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header["CROWDSAP"] = 0.99
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


@pytest.fixture
def mocked(monkeypatch):
    calls = {"search": 0, "download": 0}

    def search(tic, authors):
        calls["search"] += 1
        return INV

    def download(uri, dest, timeout=300):
        calls["download"] += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        fake_fits(dest)

    monkeypatch.setattr(f, "search_mast", search)
    monkeypatch.setattr(f, "_download", download)
    return calls


def test_dedupe_keeps_one_per_author_sector_cadence():
    rows = f.dedupe(INV)
    keys = [(r["author"], r["sector"], r["exptime"]) for r in rows]
    assert len(keys) == len(set(keys)) == 4
    qlp = [r for r in rows if r["author"] == "QLP"][0]
    assert qlp["productFilename"].endswith("v2_llc.fits")          # newest version wins


def test_pick_best_one_per_sector_prefers_spoc_120():
    best = {r["sector"]: r for r in f.dedupe(INV, pick="best")}
    assert set(best) == {1, 27}
    assert best[27]["author"] == "SPOC" and best[27]["exptime"] == 120.0


def test_exptime_filter():
    assert {r["exptime"] for r in f.dedupe(INV, exptimes=[120])} == {120.0}


def test_second_call_hits_cache_with_identical_hash(mocked, tmp_path):
    _, m1 = f.fetch(7, cache_dir=tmp_path, load=False)
    _, m2 = f.fetch(7, cache_dir=tmp_path, load=False)
    assert m1["downloaded"] == 4 and m2["downloaded"] == 0 and m2["cache_hits"] == 4
    assert mocked["search"] == 1                                   # inventory reused too
    assert m1["manifest_hash"] == m2["manifest_hash"]
    p = m1["products"][0]
    assert p["crowdsap"] == 0.99 and p["n_good"] == 450 and p["frac_good"] == 0.9
    assert {"tic", "sector", "author", "cadence_s", "product_uri", "local_path"} <= set(p)


def test_manifest_csv_written(mocked, tmp_path):
    _, m = f.fetch(7, authors=["SPOC"], cache_dir=tmp_path, load=False)
    csv = [p for p in m["manifest_paths"] if p.endswith(".csv")][0]
    lines = open(csv).read().splitlines()
    assert lines[0].startswith("tic,sector,author,cadence_s,product_uri") and len(lines) == 4


def test_mast_down_uses_stale_inventory(mocked, tmp_path, monkeypatch):
    f.fetch(7, cache_dir=tmp_path, load=False)

    def boom(tic, authors):
        raise ConnectionError("MAST 503")

    monkeypatch.setattr(f, "search_mast", boom)
    _, m = f.fetch(7, cache_dir=tmp_path, load=False, refresh=True)
    assert m["inventory"]["7"]["source"] == "stale_cache" and "MAST 503" in m["inventory"]["7"]["error"]


def test_mast_down_without_cache_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "search_mast", lambda t, a: (_ for _ in ()).throw(ConnectionError("down")))
    with pytest.raises(RuntimeError):
        f.fetch(7, cache_dir=tmp_path, load=False)


def test_download_failure_is_recorded_not_hidden(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "search_mast", lambda t, a: INV[:1])
    monkeypatch.setattr(f, "_download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("403")))
    _, m = f.fetch(7, cache_dir=tmp_path, load=False)
    assert not m["complete"] and m["failures"][0]["sector"] == 1


def test_archive_url():
    u = f.archive_url("mast:TESS/product/tess2018206045859-s0001-0000000261136679-0120-s_lc.fits")
    assert u == ("https://archive.stsci.edu/missions/tess/tid/s0001/0000/0002/6113/6679/"
                 "tess2018206045859-s0001-0000000261136679-0120-s_lc.fits")
    assert f.archive_url("mast:HLSP/tess-spoc/s0001/x.fits") == "https://archive.stsci.edu/hlsps/tess-spoc/s0001/x.fits"


@pytest.mark.live
def test_live_pi_men_multisector_spoc_2min(tmp_path):
    lcc, m1 = f.fetch(261136679, authors=["SPOC"], exptimes=[120], cache_dir=tmp_path)
    _, m2 = f.fetch(261136679, authors=["SPOC"], exptimes=[120], cache_dir=tmp_path, load=False)
    sectors = {p["sector"] for p in m1["products"]}
    assert {1, 4, 8, 11, 12, 13} <= sectors and len(lcc) == len(sectors)
    assert m2["downloaded"] == 0 and m1["manifest_hash"] == m2["manifest_hash"]
