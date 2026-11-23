"""eclipse-timing: do the transits or eclipses arrive on time? (TTVs / ETVs: the signature of an unseen body)

    eclipse-timing 141618383 --period 10.43181 --epoch 2758.99727 --duration 1.0 --depth 1386

1. Shape: one trapezoid (depth, T14, ingress) fitted to the phase-folded light curve.
2. Each event: only the mid-time moves. A chi2 scan over +-30% of T14 gives t_obs and a delta-chi2 = 1
   error (scaled up by sqrt(reduced chi2) when the event is noisier than white noise). Events missing
   ingress or egress, or with fewer than 5 points in transit, are skipped.
3. A weighted linear ephemeris is fitted to the times; O-C residuals and their reduced chi2 are reported.

flag "timing_variation" when there are >= 6 timed events, reduced chi2 >= 4, and the O-C spread is at least
3x the median timing error. A planet with TTVs suggests another planet; an eclipsing binary with ETVs
suggests a third body. A flag is a lead for follow-up, not a detection: spots, detrending and
blends also shift times.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .common import jsonable, parse_tic

MIN_EVENTS, CHI2_FLAG, SPREAD_FACTOR = 6, 4.0, 3.0


def _fold_shape(secs, P, T0, dur_d, depth):
    from .duo_solver import fit_event

    t = np.concatenate([s["time"][s["good"] & np.isfinite(s["flat"])] for s in secs])
    f = np.concatenate([s["flat"][s["good"] & np.isfinite(s["flat"])] for s in secs])
    tf = ((t - T0 + 0.5 * P) % P) - 0.5 * P + T0
    o = np.argsort(tf)
    r = fit_event(tf[o], f[o], T0, dur_d, depth)
    return r


def time_event(t, f, tc, depth, t14, tau, cad, sig):
    from .duo_solver import binned_model

    w = np.abs(t - tc) <= 1.5 * t14
    t, f = t[w], f[w]
    inn = np.abs(t - tc) <= 0.5 * t14
    if inn.sum() < 5 or not ((t < tc - 0.5 * t14).any() and (t > tc + 0.5 * t14).any()):
        return None
    # ingress and egress both need data
    if not ((np.abs(t - (tc - 0.5 * t14 + 0.5 * tau)) <= max(tau, cad)).any() and
            (np.abs(t - (tc + 0.5 * t14 - 0.5 * tau)) <= max(tau, cad)).any()):
        return None
    grid = tc + np.linspace(-0.3 * t14, 0.3 * t14, 241)
    chi = np.array([np.sum(((f - binned_model(t, g, depth, t14, tau, cad)) / sig) ** 2) for g in grid])
    i = int(np.argmin(chi))
    if i in (0, len(grid) - 1):
        return None
    lo, hi = max(i - 2, 0), min(i + 3, len(grid))
    a, b, _ = np.polyfit(grid[lo:hi] - grid[i], chi[lo:hi], 2)
    best = grid[i] - (b / (2 * a) if a > 0 else 0.0)
    scale = max(1.0, chi[i] / max(len(t) - 1, 1))
    err = np.sqrt(scale / a) if a > 0 else np.nan
    # delta-chi2 = 1 from the scan itself when the parabola is unreliable
    if not np.isfinite(err) or err <= 0:
        above = grid[chi <= chi[i] + scale]
        err = 0.5 * (above.max() - above.min()) if len(above) > 1 else np.nan
    return {"t_obs_btjd": float(best), "t_err_d": float(err), "n_points": int(len(t))}


def timing(tic, period_d, epoch_btjd, duration_h, depth_ppm, lc_files=None, cache_dir=None, secs=None, max_events=80):
    from .allsector_gate import load_sectors, robust_std
    from .duo_solver import prepare

    tic = parse_tic(tic)
    P, T0, dur_d = float(period_d), float(epoch_btjd), float(duration_h) / 24
    if secs is None:
        secs, _ = load_sectors(tic, cache_dir, lc_files)
    tmin = min(np.nanmin(s["time"]) for s in secs)
    tmax = max(np.nanmax(s["time"]) for s in secs)
    ns = np.arange(np.ceil((tmin - T0) / P), np.floor((tmax - T0) / P) + 1).astype(int)
    prepare(secs, list(T0 + ns * P), dur_d)
    shape = _fold_shape(secs, P, T0, dur_d, max(depth_ppm, 10.0) * 1e-6)
    out = {"tool": "eclipse_timing", "tic": tic, "period_d": P, "epoch_btjd": T0, "shape": {
        k: shape.get(k) for k in ("status", "depth_ppm", "t14_h", "ingress_frac", "shape")}}
    if shape.get("status") != "ok":
        return {**out, "flag": None, "result": "incomplete", "reason": "fold fit failed"}
    depth, t14 = shape["depth_ppm"] * 1e-6, shape["t14_h"] / 24
    tau = shape["ingress_frac"] * t14
    rows = []
    for s in secs:
        ok = s["good"] & np.isfinite(s["flat"])
        t, f = s["time"][ok], s["flat"][ok]
        if len(t) < 20:
            continue
        cad = float(np.median(np.diff(t)))
        sig = robust_std(f)
        for n in ns:
            tc = T0 + n * P
            if t.min() - t14 <= tc <= t.max() + t14:
                r = time_event(t, f, tc, depth, t14, tau, cad, sig)
                if r and np.isfinite(r["t_err_d"]):
                    rows.append({"epoch": int(n), "sector": s["sector"], **r})
    rows = sorted(rows, key=lambda r: r["epoch"])[:max_events]
    out["n_events"] = len(rows)
    if len(rows) < 3:
        return {**out, "flag": None, "result": "incomplete", "reason": f"{len(rows)} timed events", "events": rows}
    n = np.array([r["epoch"] for r in rows], float)
    to = np.array([r["t_obs_btjd"] for r in rows])
    e = np.array([r["t_err_d"] for r in rows])
    w = 1 / e ** 2
    A = np.vstack([np.ones_like(n), n]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    t0f, pf = cov @ (A.T @ (w * to))
    oc = to - (t0f + pf * n)
    chi2r = float(np.sum((oc / e) ** 2) / max(len(rows) - 2, 1))
    spread = float(1.4826 * np.median(np.abs(oc - np.median(oc))))
    med_err = float(np.median(e))
    for r, x in zip(rows, oc):
        r["o_minus_c_min"] = round(float(x) * 1440, 2)
        r["err_min"] = round(r["t_err_d"] * 1440, 2)
    flagged = len(rows) >= MIN_EVENTS and chi2r >= CHI2_FLAG and (np.max(np.abs(oc)) >= SPREAD_FACTOR * med_err)
    out.update(result="flag" if flagged else "pass", flag="timing_variation" if flagged else None,
               period_fit_d=round(float(pf), 7), period_fit_err_d=float(np.sqrt(cov[1, 1])),
               epoch_fit_btjd=round(float(t0f), 5), chi2_red=round(chi2r, 2), oc_spread_min=round(spread * 1440, 2),
               oc_max_min=round(float(np.max(np.abs(oc))) * 1440, 2), median_err_min=round(med_err * 1440, 2),
               events=rows)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tic")
    ap.add_argument("--period", type=float, required=True)
    ap.add_argument("--epoch", type=float, required=True)
    ap.add_argument("--duration", type=float, required=True, help="hours")
    ap.add_argument("--depth", type=float, required=True, help="ppm")
    ap.add_argument("--lc-file", action="append")
    ap.add_argument("--cache-dir")
    ap.add_argument("--full", action="store_true", help="print every event")
    a = ap.parse_args(argv)
    r = timing(a.tic, a.period, a.epoch, a.duration, a.depth, a.lc_file, a.cache_dir)
    if not a.full:
        r = {k: v for k, v in r.items() if k != "events"}
    print(json.dumps(jsonable(r), indent=None if not a.full else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
