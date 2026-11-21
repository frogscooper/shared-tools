"""allsector_gate on a committed WASP-62 b fixture and on synthetic light curves."""
import numpy as np
import pandas as pd
import pytest

from shared_tools.exo.allsector_gate import gate

WASP62 = dict(tic=149603524, period_d=4.411937227697335, epoch_btjd=3982.065462, duration_h=3.632)


def synth(path, sectors, period, t0, dur_h, depth_fn, noise=2e-4, seed=1):
    """Write a fake multi-sector CSV. depth_fn(epoch, sector) -> fractional depth."""
    rng = np.random.default_rng(seed)
    rows = []
    for sec, start in sectors:
        t = np.arange(start, start + 26.0, 10 / 1440)
        t = t[(t < start + 12.5) | (t > start + 13.5)]           # mid-sector downlink gap
        f = 1 + rng.normal(0, noise, len(t))
        n = np.round((t - t0) / period).astype(int)
        intr = np.abs(t - (t0 + n * period)) <= dur_h / 48
        for k in np.unique(n[intr]):
            sel = intr & (n == k)
            f[sel] -= depth_fn(k, sec)
        rows.append(pd.DataFrame({"sector": sec, "author": "SPOC", "cadence_s": 600.0, "time": t, "flux": f,
                                  "flux_err": noise, "quality": 0}))
    pd.concat(rows).to_csv(path, index=False)
    return path


SECTORS = [(1, 1000.0), (2, 1027.0), (3, 1054.0)]
P, T0, DUR = 3.3, 1001.1, 3.0


def run(tmp_path, csv, **kw):
    return gate(9, kw.pop("period", P), T0, DUR, lc_files=[csv], out_dir=tmp_path / "out", plot=False, **kw)


def test_wasp62_control_passes_with_matching_odd_even(fixtures, tmp_path):
    r = gate(**WASP62, lc_files=[fixtures / "wasp62_s1_s2_10min.csv.gz"], out_dir=tmp_path)
    t = r["tests"]
    assert r["verdict"] == "pass", r["reasons"]
    assert t["repeats"]["repeating"] and t["repeats"]["detected_sectors"] == [1, 2]
    assert t["odd_even"]["result"] == "pass" and t["odd_even"]["diff_sigma"] < 3
    assert t["missing_transit"]["n_missing"] == 0
    assert 12000 < r["reference_depth_ppm"] < 16000          # published depth ~1.4-1.5%
    assert (tmp_path / "gate.png").exists() and (tmp_path / "gate.json").exists()
    ev = r["events"][0]
    for k in ("sector", "depth_ppm", "depth_err_ppm", "n_in", "n_oot", "flags"):
        assert k in ev
    assert set(ev["flags"]) >= {"missing_coverage", "quality_bits_in_window", "stray_light", "momentum_dump", "near_edge"}


def test_clean_periodic_signal_passes(tmp_path):
    r = run(tmp_path, synth(tmp_path / "a.csv", SECTORS, P, T0, DUR, lambda k, s: 2e-3))
    assert r["verdict"] == "pass", r["reasons"]


def test_missing_transit_fails(tmp_path):
    csv = synth(tmp_path / "a.csv", SECTORS, P, T0, DUR, lambda k, s: 0.0 if s == 3 else 2e-3)
    r = run(tmp_path, csv)
    assert r["tests"]["missing_transit"]["result"] == "fail"
    assert {m["sector"] for m in r["tests"]["missing_transit"]["missing"]} == {3}
    assert r["verdict"] == "fail"


def test_odd_even_mismatch_is_hard_fail(tmp_path):
    csv = synth(tmp_path / "a.csv", SECTORS, P, T0, DUR, lambda k, s: 3e-3 if k % 2 else 1.5e-3)
    r = run(tmp_path, csv, depth_ppm=None)
    assert r["tests"]["odd_even"]["result"] == "fail" and r["verdict"] == "fail"


def test_odd_even_fails_closed_without_both_parities(tmp_path):
    # period so long that only one transit falls in the data -> odd/even cannot be tested
    csv = synth(tmp_path / "a.csv", [(1, 1000.0)], 40.0, T0, DUR, lambda k, s: 2e-3)
    r = gate(9, 40.0, T0, DUR, lc_files=[csv], out_dir=tmp_path / "o", plot=False)
    assert r["tests"]["odd_even"]["result"] == "incomplete"
    assert r["verdict"] != "pass"


def test_events_only_in_claimed_sector_do_not_repeat(tmp_path):
    csv = synth(tmp_path / "a.csv", SECTORS, P, T0, DUR, lambda k, s: 2e-3 if s == 2 else 0.0)
    r = run(tmp_path, csv, claimed_sectors=[2], depth_ppm=2000)
    assert r["tests"]["repeats"]["repeating"] is False
    assert r["tests"]["repeats"]["result"] == "fail" and r["verdict"] == "fail"


def test_harmonic_period_is_hard_fail(tmp_path):
    csv = synth(tmp_path / "a.csv", SECTORS + [(4, 1081.0)], 13.622518, T0, DUR, lambda k, s: 2e-3)
    r = gate(9, 13.622518, T0, DUR, lc_files=[csv], out_dir=tmp_path / "o", plot=False)
    assert r["tests"]["orbit_harmonic"]["result"] == "fail" and r["verdict"] == "fail"


def test_secondary_eclipse_fails(tmp_path):
    def depth(k, s):
        return 2e-3
    csv = synth(tmp_path / "a.csv", SECTORS, P, T0, DUR, depth)
    df = pd.read_csv(csv)
    ph = ((df.time - T0) / P) % 1
    df.loc[np.abs(ph - 0.5) * P <= DUR / 48, "flux"] -= 1e-3            # 50% secondary
    df.to_csv(csv, index=False)
    r = run(tmp_path, csv)
    assert r["tests"]["secondary"]["result"] == "fail"


def test_idempotent_output(tmp_path):
    csv = synth(tmp_path / "a.csv", SECTORS, P, T0, DUR, lambda k, s: 2e-3)
    a, b = run(tmp_path, csv), run(tmp_path, csv)
    assert a["events"] == b["events"] and a["tests"] == b["tests"]


# ---------------------------------------------------------------- live / cached real data
@pytest.mark.live
def test_live_288183107_rejected_for_missing_s73_transit():
    r = gate(288183107, 12.963929714582635, 3319.20706782588, 6.208457815172235, depth_ppm=2140,
             claimed_sectors=[74, 75], plot=False)
    miss = r["tests"]["missing_transit"]["missing"]
    assert r["verdict"] == "fail" and any(m["sector"] == 73 for m in miss)


@pytest.mark.live
def test_live_154566531_not_repeating():
    r = gate(154566531, 13.622518167549904, 3317.833110750544, 4.1936721785510676, depth_ppm=3049,
             claimed_sectors=[74, 75], plot=False)
    assert r["tests"]["repeats"]["repeating"] is False and r["verdict"] == "fail"


@pytest.mark.live
def test_live_pi_men_c_passes():
    r = gate(261136679, 6.2678139196372324, 3901.578202, 2.789, plot=False)
    assert r["verdict"] == "pass" and len(r["tests"]["repeats"]["detected_sectors"]) >= 10
