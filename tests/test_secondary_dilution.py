import numpy as np
import pandas as pd
import pytest

from shared_tools.exo import secondary_dilution as sd

WASP62 = dict(period_d=4.411937227697335, epoch_btjd=3982.065462, duration_h=3.632)


def synth(path, sec_depth=0.0, sec_phase=0.5, P=3.3, T0=1001.1, dur_h=3.0, depth=3e-3, noise=3e-4, seed=2):
    rng = np.random.default_rng(seed)
    t = np.arange(1000.0, 1081.0, 10 / 1440)
    f = 1 + rng.normal(0, noise, len(t))
    ph = ((t - T0) / P) % 1
    f[np.abs(((ph + 0.5) % 1) - 0.5) * P <= dur_h / 48] -= depth
    f[np.abs(ph - sec_phase) * P <= dur_h / 48] -= sec_depth
    sector = 1 + ((t - 1000.0) // 27).astype(int)
    pd.DataFrame({"sector": sector, "author": "SPOC", "cadence_s": 600.0, "time": t, "flux": f, "flux_err": noise,
                  "quality": 0}).to_csv(path, index=False)
    return path


def test_injected_secondary_at_phase_half_recovered(tmp_path):
    csv = synth(tmp_path / "a.csv", sec_depth=600e-6)
    r = sd.run(9, 3.3, 1001.1, 3.0, neighbors=[], lc_files=[csv], out_dir=tmp_path / "o", plot=False)
    p = r["secondary"]["phase_0p5"]
    assert r["flags"]["secondary_at_0p5"] and p["depth_ppm"] == pytest.approx(600, rel=0.25)


def test_injected_eccentric_secondary_found_by_scan(tmp_path):
    csv = synth(tmp_path / "a.csv", sec_depth=800e-6, sec_phase=0.37)
    r = sd.run(9, 3.3, 1001.1, 3.0, neighbors=[], lc_files=[csv], out_dir=tmp_path / "o", plot=False)
    best = r["secondary"]["strongest_other_phase"]
    assert best["phase"] == pytest.approx(0.37, abs=0.02) and r["flags"]["dip_elsewhere_in_phase"]


def test_no_secondary_in_clean_signal(tmp_path):
    r = sd.run(9, 3.3, 1001.1, 3.0, neighbors=[], lc_files=[synth(tmp_path / "a.csv")], out_dir=tmp_path / "o",
               plot=False)
    assert not r["flags"]["secondary_at_0p5"]


def test_bright_neighbor_flagged_faint_not():
    nb = [{"source_id": 1, "sep_arcsec": 0.0, "delta_g": 0.0, "is_target": True},
          {"source_id": 2, "sep_arcsec": 30.0, "delta_g": 2.0, "is_target": False},     # 13.7% of light
          {"source_id": 3, "sep_arcsec": 40.0, "delta_g": 10.0, "is_target": False},    # 0.01% of light
          {"source_id": 4, "sep_arcsec": 200.0, "delta_g": 1.0, "is_target": False}]    # bright but far
    rows = {r["source_id"]: r for r in sd.dilution_table(nb, 3000.0)}
    assert 1 not in rows
    assert rows[2]["flux_fraction"] == pytest.approx(1 / (1 + 10 ** 0.8), rel=1e-6)
    assert rows[2]["needed_eclipse_depth"] == pytest.approx(3000e-6 * (1 + 10 ** 0.8), rel=1e-4)
    assert rows[2]["could_cause"] and not rows[3]["could_cause"] and not rows[4]["could_cause"]


def test_wasp62_fixture_has_no_significant_secondary(fixtures, tmp_path):
    r = sd.run(149603524, **WASP62, neighbors=[], lc_files=[fixtures / "wasp62_s1_s2_10min.csv.gz"],
               out_dir=tmp_path, plot=False)
    assert not r["flags"]["secondary_at_0p5"] and abs(r["secondary"]["phase_0p5"]["snr"]) < 3
    assert r["observed_depth_ppm"] == pytest.approx(13200, rel=0.1)


@pytest.mark.live
def test_live_wasp62_all_sectors_no_secondary():
    r = sd.run(149603524, **WASP62, plot=False)
    assert not r["flags"]["secondary_at_0p5"] and not r["flags"]["neighbor_could_cause"]
