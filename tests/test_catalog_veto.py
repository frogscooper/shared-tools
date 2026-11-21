"""catalog_veto with every network call mocked, plus opt-in live regressions."""
import pytest
import requests

from shared_tools.exo import catalog_veto as cv

WASP62 = {"ra": 87.1399740872908, "dec": -63.988440787086, "tmag": 9.7109, "gaia_dr2_id": 4756649415309914240}
GAIA_HEADER = "source_id,ra,dec,pmra,pmdec,phot_g_mean_mag,parallax,ruwe\n"
GAIA_EB_HEADER = "source_id,frequency,ra,dec,pmra,pmdec,phot_g_mean_mag\n"


class FakeResp:
    def __init__(self, text, status=200, url=""):
        self.text, self.status_code, self.content, self.url = text, status, text.encode(), url


def make_router(fixtures, nasa=None, toi=None, prsa=None):
    toi_text = toi if toi is not None else (fixtures / "toi_sample.csv").read_text()
    ctoi_text = (fixtures / "ctoi_sample.csv").read_text()
    nasa_resp = nasa or FakeResp((fixtures / "nasa_wasp62.csv").read_text())
    prsa_text = prsa if prsa is not None else (fixtures / "vizier_prsa_empty.tsv").read_text()
    gaia_rows = f"{WASP62['gaia_dr2_id']},{WASP62['ra']},{WASP62['dec']},-15.6,26.0,10.077,5.69,0.97\n"

    def fake_request(method, url, headers=None, timeout=None, params=None, data=None, **kw):
        if "download_toi" in url:
            return toi_text if isinstance(toi_text, FakeResp) else FakeResp(toi_text)
        if "download_ctoi" in url:
            return FakeResp(ctoi_text)
        if "exoplanetarchive" in url:
            return nasa_resp
        if "vizier" in url:
            return FakeResp(prsa_text)
        if "gea.esac" in url:
            q = data["QUERY"]
            return FakeResp(GAIA_EB_HEADER if "vari_eclipsing_binary" in q else GAIA_HEADER + gaia_rows)
        raise AssertionError(f"unexpected URL {url}")

    return fake_request


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(cv, "BACKOFF_S", 0.0)
    monkeypatch.setattr(cv, "tic_position", lambda tic: WASP62)
    return tmp_path


def run(monkeypatch, router, cache, tic=149603524, **kw):
    monkeypatch.setattr(requests, "request", router)
    return cv.veto(tic, cache_dir=cache, refresh=True, **kw)


def test_wasp62_is_confirmed_toi_and_known_planet(monkeypatch, offline, fixtures):
    r = run(monkeypatch, make_router(fixtures, prsa=(fixtures / "vizier_prsa_wasp62.tsv").read_text()), offline)
    assert r["flags"]["toi"] and r["flags"]["known_planet"]
    toi = r["sources"]["exofop_toi"]["matches"][0]
    assert toi["toi"] == "102.01" and toi["tfopwg_disposition"] == "KP"
    assert r["sources"]["nasa_pscomppars"]["matches"][0]["pl_name"] == "WASP-62 b"
    assert r["summary"]["verdict"] == "known_object" and not r["summary"]["clean"]
    # Prsa+2022 lists WASP-62 at twice the planet period; it is reported, with a note.
    assert r["flags"]["eb"] and r["summary"]["notes"]


def test_period_alias_against_catalog(monkeypatch, offline, fixtures):
    r = run(monkeypatch, make_router(fixtures, prsa=(fixtures / "vizier_prsa_wasp62.tsv").read_text()), offline,
            period_d=4.4119, epoch_btjd=3982.0655)
    rel = {(a["source"], a["relation"]) for a in r["period_aliases"]}
    assert ("exofop_toi", "same") in rel and ("prsa_tess_eb", "1/2x") in rel


def test_nasa_503_is_down_and_never_clean(monkeypatch, offline, fixtures):
    html = "<html><head><title>NASA ExoPlanet Archive | Under Maintenance</title></head></html>"
    router = make_router(fixtures, nasa=FakeResp(html, 503))
    r = run(monkeypatch, router, offline, tic=105755855)   # a TIC with no catalog matches
    nasa = r["sources"]["nasa_pscomppars"]
    assert nasa["status"] == "down" and "503" in nasa["error"] and nasa["failed_at"]
    assert nasa["matches"] == []                    # no fake "not in archive"
    assert r["summary"]["verdict"] == "incomplete"
    assert r["summary"]["clean"] is False and "nasa_pscomppars" in r["summary"]["sources_down"]


def test_nasa_503_keeps_last_good_with_date(monkeypatch, offline, fixtures):
    run(monkeypatch, make_router(fixtures), offline)                          # good fetch is cached
    r = run(monkeypatch, make_router(fixtures, nasa=FakeResp("maintenance", 503)), offline)
    nasa = r["sources"]["nasa_pscomppars"]
    assert nasa["status"] == "down"
    assert nasa["last_good"]["matches"][0]["pl_name"] == "WASP-62 b" and nasa["last_good"]["fetched_at"]


def test_105755855_not_toi_ctoi_or_prsa_eb(monkeypatch, offline, fixtures):
    r = run(monkeypatch, make_router(fixtures, nasa=FakeResp("pl_name,hostname,tic_id,pl_orbper,pl_tranmid,"
                                                              "pl_trandur,pl_trandep,discoverymethod,disc_year\n")),
            offline, tic=105755855)
    f = r["flags"]
    assert not (f["toi"] or f["ctoi"] or f["eb"] or f["known_planet"])
    assert r["summary"]["verdict"] == "clean" and r["summary"]["complete"]


def test_exofop_down_falls_back_to_dated_copy_but_not_clean(monkeypatch, offline, fixtures):
    (offline / "catalogs").mkdir(parents=True)
    (offline / "catalogs" / "toi_2026-01-01.csv").write_text((fixtures / "toi_sample.csv").read_text())
    r = run(monkeypatch, make_router(fixtures, toi=FakeResp("gateway timeout", 504)), offline, tic=105755855)
    toi = r["sources"]["exofop_toi"]
    assert toi["status"] == "down" and toi["last_good"]["file"].endswith("toi_2026-01-01.csv")
    assert not r["summary"]["clean"]


def test_vizier_error_block_is_down(monkeypatch, offline, fixtures):
    bad = "#\n#INFO\tservice_protocol=ASU\n#***\tThe catalog J/ApJS/258/16 is not available\n"
    r = run(monkeypatch, make_router(fixtures, prsa=bad), offline, tic=105755855)
    assert r["sources"]["prsa_tess_eb"]["status"] == "down" and not r["summary"]["clean"]


def test_period_relation():
    assert cv.period_relation(13.0, 6.5)["relation"] == "2x"
    assert cv.period_relation(2.0, 6.0)["relation"] == "1/3x"
    assert cv.period_relation(5.0, 7.3) is None


# ---------------------------------------------------------------- live
@pytest.mark.live
def test_live_wasp62(tmp_path):
    r = cv.veto(149603524, cache_dir=tmp_path)
    assert r["flags"]["toi"] and r["flags"]["known_planet"]


@pytest.mark.live
def test_live_pi_men(tmp_path):
    r = cv.veto(261136679, period_d=6.2678, cache_dir=tmp_path)
    assert r["flags"]["known_planet"] and r["sources"]["exofop_toi"]["matches"][0]["toi"] == "144.01"


@pytest.mark.live
def test_live_105755855(tmp_path):
    r = cv.veto(105755855, cache_dir=tmp_path)
    assert not (r["flags"]["toi"] or r["flags"]["ctoi"] or r["flags"]["eb"])
