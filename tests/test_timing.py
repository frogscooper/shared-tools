"""eclipse-timing on synthetic light curves (offline): a 20-minute TTV is flagged, a clock-steady transit is not."""
import numpy as np
import pandas as pd
import pytest

from shared_tools.exo.timing import timing

from synth import make_sector

P, T0, DEPTH, T14_H = 3.7, 1001.1, 4000e-6, 3.0


def lc(tmp, ttv_min, name):
    files = []
    for k, start in enumerate((1000.0, 1027.5, 1055.0, 1082.5)):
        tr = []
        n = 0
        while T0 + n * P < start + 27:
            tc = T0 + n * P
            if tc >= start:
                tr.append((tc + ttv_min / 1440 * np.sin(2 * np.pi * tc / 40.0), DEPTH, T14_H / 24, 0.12))
            n += 1
        d = make_sector(k + 1, start, tr, cad_min=10.0, noise=500e-6, seed=k)
        p = tmp / f"{name}_{k}.csv"
        pd.DataFrame({c: d[c] for c in ("time", "flux", "flux_err", "quality")}).assign(
            sector=k + 1, author="SPOC", cadence_s=600).to_csv(p, index=False)
        files.append(str(p))
    return files


@pytest.mark.parametrize("ttv,flag", [(0.0, None), (20.0, "timing_variation")])
def test_ttv_flag(tmp_path, ttv, flag):
    r = timing(1, P, T0, T14_H, DEPTH * 1e6, lc_files=lc(tmp_path, ttv, f"t{ttv}"))
    assert r["n_events"] >= 15
    assert r["flag"] == flag, {k: r.get(k) for k in ("chi2_red", "oc_max_min", "median_err_min")}
    assert r["period_fit_d"] == pytest.approx(P, abs=2e-3)
