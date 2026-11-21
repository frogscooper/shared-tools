"""transit_search on synthetic light curves (offline) and pi Men c (live)."""
import numpy as np
import pandas as pd
import pytest

from shared_tools.exo.transit_search import period_grid, search


def synth(path, n_sectors=2, noise=5e-4, period=None, t0=None, dur_h=3.0, depth=0.0, single=None, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_sectors):
        start = 1000.0 + 27.4 * k
        t = np.arange(start, start + 26.0, 2 / 1440)
        t = t[(t < start + 12.6) | (t > start + 13.6)]
        f = 1 + rng.normal(0, noise, len(t)) + 3e-4 * np.sin(2 * np.pi * t / 5.0)   # slow stellar wiggle
        if period:
            ph = ((t - t0 + 0.5 * period) % period) - 0.5 * period
            f[np.abs(ph) <= dur_h / 48] -= depth
        if single:
            tc, sd, sdur = single
            f[np.abs(t - tc) <= sdur / 48] -= sd
        rows.append(pd.DataFrame({"sector": k + 1, "author": "SPOC", "cadence_s": 120.0, "time": t, "flux": f,
                                  "flux_err": noise, "quality": 0}))
    pd.concat(rows).to_csv(path, index=False)
    return path


def test_recovers_injected_periodic(tmp_path):
    csv = synth(tmp_path / "p.csv", period=3.7, t0=1001.2, depth=1500e-6)
    r = search(1, lc_files=[csv], pmin=1.0, pmax=12, n_signals=1, out_dir=tmp_path / "o")
    top = r["signals"][0]
    assert top["kind"] == "periodic" and top["status"] == "unpromoted"
    assert top["period_d"] == pytest.approx(3.7, rel=2e-3)
    assert top["depth_ppm"] == pytest.approx(1500, rel=0.25)
    assert 2.0 <= top["duration_h"] <= 4.0
    assert (tmp_path / "o" / "signals.csv").exists()


def test_recovers_injected_single_transit(tmp_path):
    csv = synth(tmp_path / "s.csv", single=(1033.37, 2000e-6, 6.0))
    r = search(1, lc_files=[csv], pmin=1.0, pmax=12, n_signals=1, out_dir=tmp_path / "o")
    singles = [s for s in r["signals"] if s["kind"] == "single"]
    assert singles, r["signals"]
    best = max(singles, key=lambda s: s["snr"])
    assert best["epoch_btjd"] == pytest.approx(1033.37, abs=0.1)
    assert best["depth_ppm"] == pytest.approx(2000, rel=0.3)
    assert best["status"] == "unpromoted"


def test_harmonic_flag_on_signal(tmp_path):
    csv = synth(tmp_path / "h.csv", n_sectors=3, period=13.65, t0=1003.0, depth=3000e-6, dur_h=4.0)
    r = search(1, lc_files=[csv], pmin=2.0, pmax=30, n_signals=1, out_dir=tmp_path / "o")
    top = r["signals"][0]
    assert top["period_d"] == pytest.approx(13.65, rel=5e-3) and top["harmonic_flag"]


def test_period_grid_is_geometric():
    g = period_grid(1.0, 10.0, 100.0, 0.05)
    assert g[0] == 1.0 and g[-1] >= 10.0 and np.allclose(g[1:] / g[:-1], g[1] / g[0])


@pytest.mark.live
def test_live_pi_men_c(tmp_path):
    r = search(261136679, sectors=[1, 4, 8, 11, 12, 13], pmin=1, pmax=30, n_signals=1, out_dir=tmp_path)
    top = r["signals"][0]
    assert top["period_d"] == pytest.approx(6.2678, abs=0.002)
    assert 200 < top["depth_ppm"] < 400 and 2.0 <= top["duration_h"] <= 4.0


@pytest.mark.live
def test_live_injected_single_in_pi_men_s1(tmp_path):
    from shared_tools.exo.allsector_gate import load_sectors

    secs, _ = load_sectors(261136679)
    s1 = [s for s in secs if s["sector"] == 1][0]
    f = s1["flux"] / np.nanmedian(s1["flux"])
    tc, depth, dur_h = 1330.0, 1000e-6, 5.0          # between pi Men c transits (1325.50, 1331.77)
    f = np.where(np.abs(s1["time"] - tc) <= dur_h / 48, f - depth, f)
    pd.DataFrame({"sector": 1, "author": "SPOC", "cadence_s": 120.0, "time": s1["time"], "flux": f,
                  "flux_err": 1e-4, "quality": s1["quality"]}).to_csv(tmp_path / "inj.csv", index=False)
    r = search(261136679, lc_files=[tmp_path / "inj.csv"], pmin=1, pmax=12, n_signals=1, out_dir=tmp_path / "o")
    hits = [s for s in r["signals"] if s["kind"] == "single" and abs(s["epoch_btjd"] - tc) < 0.1]
    assert hits and hits[0]["depth_ppm"] == pytest.approx(1000, rel=0.3)


def test_binned_flat_keeps_ffi_cadence_points():
    """30-min FFI data binned at 10 min used to leave one point per bin, and every bin was dropped."""
    import numpy as np

    from shared_tools.exo.transit_search import binned_flat

    t = np.arange(0, 10, 30 / 1440)
    sec = {"sector": 9, "time": t, "flat": np.ones(len(t)), "good": np.ones(len(t), bool)}
    tb, fb, eb, sb = binned_flat([sec], 10.0)
    assert len(tb) == len(t)
    t2 = np.arange(0, 10, 2 / 1440)
    tb2, *_ = binned_flat([{"sector": 9, "time": t2, "flat": np.ones(len(t2)), "good": np.ones(len(t2), bool)}], 10.0)
    assert len(tb2) == pytest.approx(len(t2) / 5, rel=0.01)
