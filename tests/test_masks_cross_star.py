import numpy as np
import pandas as pd
from pathlib import Path

from shared_tools.exo.cross_star_veto import cross_star_veto, dips_from_lightcurves, flag_events, run
from shared_tools.exo.masks import DESAT, cadence_mask

from synth import make_sector

FIX = Path(__file__).parent / "fixtures"


def test_cadence_mask_reasons():
    t = np.arange(0, 10, 0.01)
    q = np.zeros(len(t), int)
    q[100] = DESAT          # momentum dump at t=1.0
    q[300] = 4096           # scattered light
    q[400] = 128            # manual exclude (default bad bit)
    t = np.r_[t[:500], t[600:]]  # 1-day gap 5.0 -> 6.0 splits two orbits
    q = np.r_[q[:500], q[600:]]
    good, why = cadence_mask(t, q, edge_d=1.0, dump_pad_d=0.05)
    assert why["orbit_edge"][t < 1.0].all() and why["orbit_edge"][(t > 4.0) & (t < 5.0)].all()
    assert why["orbit_edge"][(t >= 6.0) & (t < 7.0)].all() and why["orbit_edge"][t > 9.0].all()
    assert not why["orbit_edge"][(t > 2) & (t < 3.9)].any()
    assert why["momentum_dump"][np.abs(t - 1.0) <= 0.05].all() and 9 <= why["momentum_dump"].sum() <= 11
    assert why["scattered"][300] and why["quality"][400]
    assert not good[300] and not good[400] and good[np.argmin(np.abs(t - 2.5))]


def test_cadence_mask_qlp_any_flag_is_bad():
    good, why = cadence_mask(np.arange(5.0), np.array([0, 1, 0, 0, 0]), author="QLP", edge_d=0)
    assert list(good) == [True, False, True, True, True]


def test_hunter_overnight_regression_24_of_32():
    """hunter's 21 Nov run: 24 of the 32 raw flags shared an epoch (1 d) with an SNR>=8 dip on another star."""
    dips = pd.read_csv(FIX / "hunter_overnight_dips.csv")
    flags = pd.read_csv(FIX / "hunter_overnight_flags.csv").rename(columns={"epoch_btjd": "t0_btjd"})
    singles = flags[flags.search_mode == "single-transit"]
    out = flag_events(singles, dips, tol_d=1.0, max_stars=1, n_stars_total=200)
    assert int(out["common_mode"].sum()) == 24 and len(flags) == 32
    assert (out["common_mode"] == out["hunter_same_epoch"]).all()   # row-for-row agreement
    near = out.set_index("tic").loc[[230064179, 300560832, 359630365]]
    assert near["common_mode"].all()                                 # the three near-misses too


def test_hunter_ccd_table_finds_common_epochs():
    dips = pd.read_csv(FIX / "hunter_overnight_dips.csv")
    d, ep = cross_star_veto(dips, tol_d=0.5, max_stars=4, n_stars_total=200)
    assert len(ep) >= 5 and ep["n_stars"].iloc[0] >= 8
    assert d.loc[d.common_mode, "n_stars_at_epoch"].min() >= 5


def test_synthetic_ccd_shared_dip_vetoed_unique_dip_kept(tmp_path):
    lcs = []
    for i in range(12):
        tr = [(1007.3, 4000e-6, 4 / 24, 0.1)] if i < 7 else []
        if i == 0:
            tr.append((1020.6, 4000e-6, 4 / 24, 0.1))
        lcs.append(dict(make_sector(5, 1000.0, tr, noise=400e-6, seed=i), tic=1000 + i))
    s, d, ep = run(lcs=lcs, sector=5, tol_d=0.5, max_stars=2, out_dir=tmp_path)
    shared = d[(d.t0_btjd - 1007.3).abs() < 0.2]
    unique = d[(d.t0_btjd - 1020.6).abs() < 0.2]
    assert len(shared) >= 6 and shared.common_mode.all()
    assert len(unique) == 1 and not unique.common_mode.any()
    assert abs(ep.t_mid_btjd.iloc[0] - 1007.3) < 0.2 and ep.n_stars.iloc[0] >= 6
    assert (tmp_path / "common_mode_epochs.csv").exists() and (tmp_path / "mask_summary.csv").exists()


def test_max_frac_rule():
    dips = pd.DataFrame({"tic": [1, 2, 3, 4], "t0_btjd": [10.0, 10.1, 10.2, 30.0]})
    d, _ = cross_star_veto(dips, tol_d=0.5, max_stars=100, max_frac=0.02, n_stars_total=100)
    assert list(d.common_mode) == [True, True, True, False]
