import pytest

from shared_tools.exo import duo_solver as ds

from synth import make_sector

D, T14, TF = 3000e-6, 6 / 24, 0.1


def sectors(extra=(), v_shape=False, depth2=D, s3_start=1300.0):
    t1, t2 = 1010.0, 1710.0                    # true P = 70 d, n = 10
    tf1 = 0.5 if v_shape else TF
    return [make_sector(1, 1000.0, [(t1, D, T14, tf1)]),
            make_sector(2, 1027.5, []),
            make_sector(3, s3_start, list(extra)),
            make_sector(4, 1700.0, [(t2, depth2, T14, TF)])], t1, t2


def test_true_alias_survives_and_covered_aliases_are_excluded(tmp_path):
    secs, t1, t2 = sectors()
    r = ds.solve(1, t1 + 0.01, t2 - 0.01, 6.0, 3000, pmin=10, secs=secs, out_dir=tmp_path, rho_solar=1.0)
    by_n = {a["n"]: a for a in r["aliases"]}
    assert by_n[10]["status"] == "open" and 70.0 == pytest.approx(by_n[10]["period_d"], abs=0.01)
    assert by_n[20]["status"] == "excluded"          # P = 35 predicts 1045.0 in sector 2, which is flat
    w = [x for x in by_n[20]["windows"] if x["verdict"] == "excluded"][0]
    assert w["depth_err_ppm"] > 0 and (2800 - w["depth_ppm"]) / w["depth_err_ppm"] > 3   # a measured test
    assert r["n_excluded"] >= 20 and 70.0 == pytest.approx(min(r["surviving_periods"], key=lambda p: abs(p - 70)), abs=0.01)
    assert r["gate"]["tests"]["events_consistent"]["result"] == "pass"
    assert r["label"] == "duo candidate"
    assert by_n[10]["next_windows"] and by_n[10]["expected_t14_h_b0"] > 0
    assert (tmp_path / "aliases.csv").exists()


def test_v_shaped_event_fails_gate_a(tmp_path):
    secs, t1, t2 = sectors(v_shape=True)
    r = ds.solve(1, t1, t2, 6.0, 3000, pmin=10, secs=secs, out_dir=tmp_path)
    a = r["gate"]["tests"]["events_consistent"]
    assert a["result"] == "fail" and any("event 1" in w for w in a["why"])
    assert r["gate"]["verdict"] == "fail"


def test_depth_mismatch_fails_gate_a(tmp_path):
    secs, t1, t2 = sectors(depth2=1200e-6)
    r = ds.solve(1, t1, t2, 6.0, 3000, pmin=10, secs=secs, out_dir=tmp_path)
    assert any("depths differ" in w for w in r["gate"]["tests"]["events_consistent"]["why"])


def test_observed_third_transit_changes_label(tmp_path):
    # sector 3 now covers 1290 = t1 + 4 * 70, and the transit is there
    secs, t1, t2 = sectors(extra=[(1290.0, D, T14, TF)], s3_start=1280.0)
    r = ds.solve(1, t1, t2, 6.0, 3000, pmin=10, secs=secs, out_dir=tmp_path)
    assert r["gate"]["tests"]["label"]["result"] == "periodic"
    seen = [x["period_d"] for x in r["gate"]["tests"]["label"]["seen_ranked"]]
    assert any(abs(p - 70.0) < 0.01 for p in seen)
    assert all(abs(280 / p - round(280 / p)) < 1e-3 for p in seen)     # every "seen" alias predicts 1290
    assert "duo candidate" not in r["label"]


def test_trapezoid_fit_recovers_event():
    s = make_sector(1, 1000.0, [(1010.0, D, T14, TF)], cad_min=2.0)
    import numpy as np

    s["good"] = np.ones(len(s["time"]), bool)
    s["flat"] = s["flux"] / np.median(s["flux"])
    e = ds.fit_event(s["time"], s["flat"], 1010.02, T14, D)
    assert e["planet_shaped"] and abs(e["t0_btjd"] - 1010.0) < 3 * e["t0_err_d"] + 1e-3
    assert e["depth_ppm"] == pytest.approx(3000, rel=0.1) and e["t14_h"] == pytest.approx(6.0, abs=0.3)


@pytest.mark.live
def test_live_toi4600c_duo():
    """TOI-4600 c (Mireles+2023, P = 482.8191 d): TESS transits in S17 and S53, 38 sectors of coverage."""
    r = ds.solve(232608943, 1785.96, 2751.5986, 10.85, 12000, pmin=10, rho_solar=1.45)
    surv = r["surviving_periods"]
    assert any(abs(p - 482.8191) < 3 * 0.0093 + 0.01 for p in surv)
    assert r["n_excluded"] >= 90 and len(surv) <= 3
    assert r["gate"]["verdict"] == "pass" and r["label"] == "duo candidate"


@pytest.mark.live
def test_live_toi5678b_duo_and_later_transits():
    """TOI-5678 b (Ulmer-Moll+2023, P = 47.73 d): S4 + S30 duo; later sectors show more transits."""
    r = ds.solve(209464063, 1424.711, 2140.665, 6.86, 2400, pmin=10, sectors=[4, 30, 31])
    assert any(abs(p - 47.73) < 0.01 for p in r["surviving_periods"]) and r["n_excluded"] > 0
    r2 = ds.solve(209464063, 1424.711, 2140.665, 6.86, 2400, pmin=10)
    assert r2["gate"]["tests"]["label"]["seen_ranked"][0]["period_d"] == pytest.approx(47.73, abs=0.01)
