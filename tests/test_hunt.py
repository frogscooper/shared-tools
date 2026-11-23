"""hunt end to end on synthetic stars (offline): duo + BLS + triage + gate, pause/resume, summary."""
import json

import pandas as pd
import pytest

from shared_tools.exo import batch_runner as br
from shared_tools.exo import hunt

from synth import make_sector


def periodic(P, t0, depth, t14_h, start, ndays=27.0):
    out, t = [], t0
    while t < start - P:
        t += P
    while t < start + ndays:
        if t >= start:
            out.append((t, depth, t14_h / 24, 0.15))
        t += P
    return out


@pytest.fixture
def stars(tmp_path, monkeypatch):
    starts = {5: 1000.0, 6: 1027.5, 20: 1300.0, 45: 1700.0}
    spec = {
        # 1: a clean 5.37 d planet-like signal; 2: a duo (one dip in S5, one in S45); 3: grazing-short 10.43 d signal
        1: lambda s, st: periodic(5.37, 1001.3, 1500e-6, 3.0, st),
        2: lambda s, st: [(1010.0, 3000e-6, 6 / 24, 0.1)] if s == 5 else ([(1710.0, 3000e-6, 6 / 24, 0.1)] if s == 45 else []),
        3: lambda s, st: periodic(10.4317, 1004.0, 1400e-6, 1.0, st),
    }
    files = {}
    for tic, f in spec.items():
        files[tic] = []
        for sec, st in starts.items():
            d = make_sector(sec, st, f(sec, st), noise=400e-6, seed=tic)
            p = tmp_path / "lc" / f"TIC{tic}_s{sec}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({k: d[k] for k in ("time", "flux", "flux_err", "quality")}).assign(
                sector=sec, author="TESS-SPOC", cadence_s=1800).to_csv(p, index=False)
            files[tic].append(str(p))
    import shared_tools.exo.tess_lc_fetch as tlf

    monkeypatch.setattr(br, "tic_params", lambda tics: pd.DataFrame(
        [{"tic": t, "tmag": 11.0, "teff": 5500, "logg": 4.5, "rstar": 1.0, "mstar": 1.0, "rho_solar": 1.0} for t in tics]))
    monkeypatch.setattr(tlf, "inventory", lambda tic, **k: ([{"sector": s, "author": "TESS-SPOC"} for s in starts], {}))
    monkeypatch.setattr(tlf, "fetch", lambda tic, **k: (None, {"products": [{"local_path": f} for f in files[int(tic)]],
                                                               "failures": []}))
    return tmp_path


def args(tmp, *extra):
    return ["run", "--run-dir", str(tmp / "run"), "--tics", "1,2,3", "--no-controls", "--no-known-filter",
            "--no-duo-packets", "--workers", "1", "--dl-threads", "2", "--cache-dir", str(tmp / "cache"), "--pmax", "20",
            "--n-signals", "1", "--tpf-sectors", "0", *extra]


def test_hunt_offline_end_to_end(stars, monkeypatch, capsys):
    import shared_tools.exo.candidate_packet as cp

    built = []
    monkeypatch.setattr(cp, "build", lambda tic, sig, *a, **k: built.append((tic, sig)) or {
        "status": "gate_passed_needs_human_review", "packet_dir": "x",
        "results": {"allsector_gate": {"verdict": "pass"}, "catalog_veto": {"verdict": "clean", "flags": {}},
                    "centroid_check": {"verdict": "not_run"}, "secondary_dilution": {"flags": {}}}})
    # a 0-second budget pauses before any work, with exit code 75 and nothing lost
    assert hunt.main(args(stars, "--budget", "0.000001")) == hunt.EXIT_PAUSED
    assert "PAUSED" in capsys.readouterr().out
    assert hunt.main(args(stars, "--budget", "0")) == 0
    out = capsys.readouterr().out
    assert "DONE" in out and "periodic candidates for review: 1" in out
    run = stars / "run"
    tri = json.loads((run / "triage.json").read_text())
    by_tic = {r["tic"]: r for r in tri["survivors"] + tri["near_misses"]}
    assert 1 in by_tic and abs(by_tic[1]["period_d"] - 5.37) < 0.01
    gates = {g["tic"]: g for g in hunt.load_stage(run, "gate")}
    assert gates[1]["verdict"] == "pass" and gates[1]["tests"]["duration_density"] == "pass"
    assert [t for t, _ in built] == [1] and built[0][1]["rho_solar"] == 1.0
    # the grazing-short star never reaches vetting: the duration cut stops it in triage or the gate
    cut3 = pd.read_csv(run / "triage.csv").query("tic == 3")
    assert cut3.empty or "duration" in str(cut3.fails.iloc[0]) or gates.get(3, {}).get("verdict") != "pass"
    fun = json.loads((run / "funnel.json").read_text())
    assert any(r["tic"] == 2 and r["verdict"] == "pass" for r in fun["duo"])
    md = (run / "summary.md").read_text()
    assert "Hunt summary" in md and len(md.splitlines()) < 80
    assert (run / "hunt-log-draft.md").read_text().startswith("# Hunt log:")
    # status is a few lines
    assert hunt.status(run) == 0
    assert len(capsys.readouterr().out.splitlines()) < 15
