import pandas as pd
import pytest

from shared_tools.exo import batch_runner as br

from synth import make_sector


def test_apply_cuts_defaults_match_team_cut():
    df = pd.DataFrame([{"tic": 1, "tmag": 10.0, "teff": 5000, "logg": 4.5},
                       {"tic": 2, "tmag": 13.5, "teff": 5000, "logg": 4.5},
                       {"tic": 3, "tmag": 10.0, "teff": 6500, "logg": 4.5},
                       {"tic": 4, "tmag": 10.0, "teff": 5000, "logg": 3.5},
                       {"tic": 5, "tmag": 7.9, "teff": 3400, "logg": None}])
    out = br.apply_cuts(df, None).set_index("tic")["cut_fail"]
    assert out[1] == "" and out[2] == "tmag" and out[3] == "teff" and out[4] == "logg" and out[5] == "tmag;teff;logg"
    assert br.CUTS["crowdsap_min"] == 0.9 and br.CUTS["early_max_sector"] == 26 and br.CUTS["late_min_sector"] == 40


def test_match_pairs_needs_years_and_similar_shape():
    dips = [{"t0_btjd": 1010, "sector": 5, "depth_ppm": 3000, "duration_h": 6, "snr": 20},
            {"t0_btjd": 1100, "sector": 7, "depth_ppm": 3000, "duration_h": 6, "snr": 20},     # too close
            {"t0_btjd": 1710, "sector": 45, "depth_ppm": 2800, "duration_h": 6, "snr": 15},
            {"t0_btjd": 1800, "sector": 48, "depth_ppm": 9000, "duration_h": 6, "snr": 30}]    # depth ratio 3
    p = br.match_pairs(dips)
    assert {(x["t1"], x["t2"]) for x in p} == {(1010, 1710), (1100, 1710)}
    assert {(x["t1"], x["t2"]) for x in br.match_pairs(dips, sector=7)} == {(1100, 1710)}


def test_merge_fragments_keeps_strongest_dip_of_one_eclipse():
    # TIC 99725941-like: one long eclipse reported as a core and two wings, then a real dip years later
    dips = [{"tic": 1, "sector": 5, "t0_btjd": 1000.00, "duration_h": 8, "depth_ppm": 9000, "snr": 40},
            {"tic": 1, "sector": 5, "t0_btjd": 999.75, "duration_h": 4, "depth_ppm": 4000, "snr": 12},
            {"tic": 1, "sector": 5, "t0_btjd": 1000.28, "duration_h": 4, "depth_ppm": 4500, "snr": 11},
            {"tic": 2, "sector": 5, "t0_btjd": 1000.10, "duration_h": 4, "depth_ppm": 4500, "snr": 11},   # other star: kept
            {"tic": 1, "sector": 48, "t0_btjd": 1800.00, "duration_h": 8, "depth_ppm": 8800, "snr": 30}]
    kept, frags = br.merge_fragments(dips)
    assert sorted((d["tic"], d["t0_btjd"]) for d in kept) == [(1, 1000.0), (1, 1800.0), (2, 1000.1)]
    assert {f["fragment_of"] for f in frags} == {1000.0}
    # without merging, the wings pair with the late dip; with it, one pair remains
    assert len(br.match_pairs([d for d in dips if d["tic"] == 1])) > 1
    assert len(br.match_pairs([d for d in kept if d["tic"] == 1])) == 1


def test_match_pairs_skips_eb_sized_pairs():
    dips = [{"t0_btjd": 1000, "sector": 5, "depth_ppm": 200000, "duration_h": 6, "snr": 90},
            {"t0_btjd": 1800, "sector": 48, "depth_ppm": 210000, "duration_h": 6, "snr": 90}]
    assert len(br.match_pairs(dips)) == 1                       # no radius known: kept
    assert br.match_pairs(dips, rstar_rsun=1.0) == []           # 20% deep on a Sun: ~50 Re


def test_cached_stage_is_not_recomputed(tmp_path):
    run = br.Run(tmp_path)
    calls = []
    f = lambda: calls.append(1) or {"ok": True, "x": 1}
    assert run.cached("dips", "TIC1", f)["x"] == 1 and run.cached("dips", "TIC1", f)["x"] == 1
    assert len(calls) == 1
    br.Run(tmp_path, redo=["dips"]).cached("dips", "TIC1", f)
    assert len(calls) == 2
    err = run.cached("duo", "bad", lambda: 1 / 0)          # an error is recorded, not raised
    assert err["ok"] is False and "ZeroDivisionError" in err["error"]


def test_end_to_end_offline_synthetic_duo(tmp_path, monkeypatch):
    import shared_tools.exo.tess_lc_fetch as tlf

    tr = (6 / 24, 0.1)
    secs = [make_sector(5, 1000.0, [(1010.0, 3000e-6, *tr)]), make_sector(6, 1027.5, []),
            make_sector(20, 1300.0, []), make_sector(45, 1700.0, [(1710.0, 3000e-6, *tr)])]
    files = []
    for s in secs:
        p = tmp_path / f"s{s['sector']}.csv"
        pd.DataFrame({k: s[k] for k in ("time", "flux", "flux_err", "quality")}).assign(
            sector=s["sector"], author="TESS-SPOC", cadence_s=1800).to_csv(p, index=False)
        files.append(str(p))
    monkeypatch.setattr(br, "tic_params", lambda tics: pd.DataFrame(
        [{"tic": 7, "tmag": 11.0, "teff": 4800, "logg": 4.6, "rho_solar": 1.5}]))
    monkeypatch.setattr(tlf, "inventory", lambda tic, **k: ([{"sector": s, "author": "TESS-SPOC"} for s in (5, 6, 20, 45)], {}))
    monkeypatch.setattr(tlf, "fetch", lambda tic, **k: (None, {"products": [{"local_path": f} for f in files],
                                                               "failures": []}))
    r = br.run_batch(tmp_path / "run", tics=[7], known_filter=False, packets=False)
    f = r["funnel"]
    assert f["n_selected"] == 1 and f["n_early_and_late_coverage"] == 1 and f["n_pairs"] >= 1
    assert f["n_duo_gate_pass"] >= 1
    duo = pd.read_csv(r["files"]["duo_pairs"])
    hit = duo[(duo.t1 - 1010).abs() < 0.3]
    assert len(hit) and "70.0" in hit.surviving_periods.iloc[0]
    assert (tmp_path / "run" / "funnel.csv").exists()
    # second call reuses every stage file
    r2 = br.run_batch(tmp_path / "run", tics=[7], known_filter=False, packets=False)
    assert r2["funnel"]["n_duo_gate_pass"] == f["n_duo_gate_pass"]


@pytest.mark.live
def test_live_control_toi4600c(tmp_path):
    r = br.run_batch(tmp_path, control=["toi4600c"], packets=False)
    c = r["controls"][0]
    assert c["pair_found"] and c["true_period_survives"] and c["duo_gate"] == "pass"
