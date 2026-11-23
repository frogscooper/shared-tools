"""survey end to end on a synthetic CCD (offline): quick tier, CCD bad window, deep tier, classify, leaderboard."""
import json

import pandas as pd
import pytest

from shared_tools.exo import batch_runner as br
from shared_tools.exo import survey

from synth import make_sector

STARTS = {5: 1000.0, 6: 1027.5, 20: 1300.0, 45: 1700.0}
QUICK_SECTOR = 45


def periodic(P, t0, depths, t14_h, start, ndays=27.0):
    """depths: one value, or (odd, even) for an EB with unequal eclipses."""
    depths = depths if isinstance(depths, tuple) else (depths, depths)
    n0 = int((start - t0) // P) - 1
    out = []
    for n in range(n0, n0 + int(ndays / P) + 3):
        t = t0 + n * P
        if start <= t < start + ndays:
            out.append((t, depths[n % 2], t14_h / 24, 0.15))
    return out


SPEC = {
    1: lambda s, st: periodic(5.37, 1001.3, 2000e-6, 3.0, st),                          # planet-like
    2: lambda s, st: periodic(2.43, 1000.7, (30000e-6, 12000e-6), 2.5, st),             # EB, unequal eclipses
    3: lambda s, st: [(1706.0, 5000e-6, 8 / 24, 0.1)] if s == QUICK_SECTOR else [],     # one deep dip
}
ROTATOR = 4
NOISE = list(range(10, 25))   # 15 quiet stars sharing one systematic dip at BTJD 1720.0


def write_star(tmp, tic, sec):
    st = STARTS[sec]
    tr = SPEC[tic](sec, st) if tic in SPEC else ([(1720.0, 2500e-6, 4 / 24, 0.1)] if sec == QUICK_SECTOR else [])
    d = make_sector(sec, st, tr, noise=400e-6, seed=tic)
    if tic == ROTATOR:   # spotted star: 0.6% two-spot modulation, the kind of signal that is not an EB
        import numpy as np

        ph = 2 * np.pi * d["time"] / 3.3
        d["flux"] = d["flux"] * (1 + 0.006 * np.sin(ph) + 0.003 * np.sin(2 * ph + 1.0))
    p = tmp / "lc" / f"TIC{tic}_s{sec}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({k: d[k] for k in ("time", "flux", "flux_err", "quality")}).assign(
        sector=sec, author="TESS-SPOC", cadence_s=1800).to_csv(p, index=False)
    return str(p)


@pytest.fixture
def ccd(tmp_path, monkeypatch):
    import shared_tools.exo.candidate_packet as cp
    import shared_tools.exo.catalog_veto as cv
    import shared_tools.exo.tess_lc_fetch as tlf

    tics = sorted(list(SPEC) + [ROTATOR] + NOISE)
    files = {t: {s: write_star(tmp_path, t, s) for s in STARTS} for t in tics}
    monkeypatch.setattr(br, "ccd_targets", lambda *a, **k: tics)
    monkeypatch.setattr(br, "tic_params", lambda ids: pd.DataFrame(
        [{"tic": t, "tmag": 11.0, "teff": 5500, "logg": 4.5, "rstar": 1.0, "mstar": 1.0, "rho_solar": 1.0,
          "ra": 10.0 + t, "dec": -60.0} for t in ids]))
    monkeypatch.setattr(survey, "_download", lambda cache, tic, secs: ([files[int(tic)][QUICK_SECTOR]], [], []))
    monkeypatch.setattr(tlf, "inventory", lambda tic, **k: ([{"sector": s, "author": "TESS-SPOC"} for s in STARTS], {}))
    monkeypatch.setattr(tlf, "fetch", lambda tic, **k: (None, {"products": [{"local_path": f} for f in files[int(tic)].values()],
                                                               "failures": []}))
    monkeypatch.setattr(cp, "build", lambda tic, sig, *a, **k: {
        "status": "gate_passed_needs_human_review", "packet_dir": f"pk/TIC{tic}",
        "results": {"allsector_gate": {"verdict": "pass"}, "catalog_veto": {"verdict": "clean", "flags": {}},
                    "centroid_check": {"verdict": "on_target"}, "secondary_dilution": {"flags": {}}}})
    monkeypatch.setattr(cv, "veto", lambda tic, **k: {"flags": {"eb": False}, "summary": {"sources_down": []}})
    return tmp_path


def test_survey_offline_end_to_end(ccd, capsys):
    d = ccd / "sv"
    assert survey.main(["plan", "--dir", str(d), "--sectors", "45", "--cameras", "1", "--ccds", "2"]) == 0
    common = ["--dir", str(d), "--workers", "1", "--dl-threads", "2", "--no-controls", "--no-known-filter",
              "--tpf-sectors", "0", "--cache-dir", str(ccd / "cache")]
    assert survey.main(["run", *common, "--budget", "0.000001"]) == survey.EXIT_PAUSED
    assert survey.main(["run", *common, "--budget", "0"]) == 0
    capsys.readouterr()
    job = d / "jobs" / "s45_c1_d2"
    fl = json.loads((job / "flagged.json").read_text())
    assert fl["n_quick_ok"] == 19
    assert any(w["start_btjd"] <= 1720.0 <= w["end_btjd"] for w in fl["bad_windows"]), fl["bad_windows"]
    flagged = {r["tic"] for r in fl["flagged"]}
    assert {1, 2, 3} <= flagged and not flagged & set(NOISE)          # the shared dip flags nobody
    assert (job / "deep" / "bad_windows_extra.json").exists()
    lb = pd.read_csv(d / "leaderboard.csv")
    cls = {int(r.tic): r["class"] for _, r in lb.iterrows()}
    assert cls.get(1) == "planet_candidate"
    assert cls.get(2) == "eb_candidate"
    assert cls.get(3) == "single_transit"
    assert ROTATOR not in cls                                         # a spotted star is a variable, not a lead
    tags = {int(r.tic): str(r.tags) for _, r in lb.iterrows()}
    assert "long_period" in tags[3] and "timing_variation" not in tags[1]   # steady planet: no TTV flag
    cl = json.loads((job / "classify.json").read_text())
    assert any(p.name.startswith("timing_TIC1_") for p in (job / "stages" / "classify").glob("*.json"))
    assert lb.iloc[0]["class"] == "planet_candidate" and lb.novelty.str.startswith("new").all()
    md = (d / "survey_summary.md").read_text()
    assert "1/1 CCDs done" in md and len(md.splitlines()) < 60
    assert survey.status(d) == 0 and survey.top(d, 5) == 0
    out = capsys.readouterr().out
    assert "planet_candidate" in out and len(out.splitlines()) < 12


def test_extra_dips_and_size_tags(tmp_path):
    bls = tmp_path / "stages" / "bls"
    bls.mkdir(parents=True)
    singles = [{"epoch_btjd": 1000.0, "duration_h": 3, "depth_ppm": 20000, "snr": 50},     # on the ephemeris
               {"epoch_btjd": 1005.0, "duration_h": 3, "depth_ppm": 400, "snr": 12},       # mid-phase: secondary
               {"epoch_btjd": 1013.3, "duration_h": 3, "depth_ppm": 900, "snr": 15},       # shallow, off-phase
               {"epoch_btjd": 1022.6, "duration_h": 3, "depth_ppm": 15000, "snr": 40}]     # deep, off-phase
    (bls / "TIC9.json").write_text(json.dumps({"singles": singles}))
    ex = survey.extra_dips(tmp_path, 9, 10.0, 1000.0, 3.0, [])
    assert [x["t"] for x in ex] == [1013.3, 1022.6]
    r = {"class": "single_transit", "depth_ppm": 10000.0}
    assert survey.size_tags(r, 1.0) == ["jupiter_bd_or_mdwarf_size", "long_period"] and r["rp_re"] == pytest.approx(10.9, abs=0.1)
    assert "stellar_companion" in survey.size_tags({"class": "eb_candidate", "depth_ppm": 150000.0, "period_d": 3.0}, 1.0)
    assert survey.size_tags({"class": "planet_candidate", "depth_ppm": 200.0, "period_d": 3.0}, 1.0) == ["small_planet"]
