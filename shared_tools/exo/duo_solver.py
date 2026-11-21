"""duo_solver: allowed periods for two single transits, checked against all TESS coverage.

Two transits at T1 < T2 allow P_n = (T2 - T1) / n for n = 1 .. floor((T2 - T1) / pmin).
For every alias, each predicted transit that falls inside existing good-quality data is measured
(allsector_gate.measure_event: local depth and SNR on the detrended flux). A window is

  excluded    the data cover it and rule out the reference depth: (ref - depth)/err >= 3 and depth/err < 3
  seen        a dip of consistent depth is there (|depth - ref|/err < 3 and depth/err >= 3)
  unresolved  in coverage, but too gappy, one-sided or flagged to decide

An alias is excluded if any window excludes it; otherwise seen > unresolved > open (no window in data).

The vetter's duo gate (hard pass/fail):
  (a) events_consistent   both depths and both T14 agree within 3 sigma (5% systematic floor on
                          depth), and each event is flat-bottomed on its own (trapezoid beats a
                          V by delta-chi2 >= 9 and ingress/T14 < 0.4)
  (b) coverage_excludes   no alias is left "unresolved": every alias that predicts a transit in
                          coverage is decided by the data (fails closed)
  (c) label               "duo candidate" until a predicted transit is observed; if one is, the label
                          says so and the signal must go through allsector_gate as a periodic signal

With --rho (solar units) or --mstar/--rstar, each alias also gets the T14 expected for a central,
circular transit, and the ratio observed/expected.

    duo-solver 232608943 --t1 1785.96 --t2 2751.5986 --duration 10.85 --depth 6928 --pmin 10 --rho 1.45
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .allsector_gate import detrend, load_sectors, measure_event, robust_std
from .common import ET, default_cache_dir, jsonable, now_et, parse_tic, write_json
from .masks import cadence_mask

SIGMA = 3.0
SHAPE_DCHI2 = 9.0
V_INGRESS_FRAC = 0.4
DEPTH_FLOOR = 0.05
BTJD0 = 2457000.0


def btjd_to_datetime(t):
    jd = t + BTJD0
    return datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(days=jd - 2451545.0)


def now_btjd():
    return (datetime.now(timezone.utc) - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400 + 2451545.0 - BTJD0


# ---------------------------------------------------------------- data
def prepare(secs, event_times, dur_d):
    """Quality mask (masks.cadence_mask without orbit-edge trimming; measure_event flags edges itself)
    and a biweight detrend with both events masked."""
    for s in secs:
        t = s["time"]
        s["good"], _ = cadence_mask(t, s["quality"], s["flux"], s["author"], edge_d=0)
        in_tr = np.zeros(len(t), bool)
        for tc in event_times:
            in_tr |= np.abs(t - tc) <= 0.75 * dur_d
        s["flat"] = detrend(t, s["flux"], s["good"], in_tr, max(0.5, 3.0 * dur_d))
    return secs


def trapezoid(t, t0, depth, t14, tau):
    """Flat-bottomed trapezoid: total duration t14, ingress/egress tau each."""
    x = np.abs(t - t0)
    tau = max(tau, 1e-6)
    y = np.clip((t14 / 2 - x) / tau, 0, 1)
    return 1 - depth * y


def binned_model(t, t0, depth, t14, tau, cadence_d):
    """Trapezoid averaged over the exposure (5 sub-samples) so 30-min FFI cadences are not biased."""
    if cadence_d < 10 / 1440:
        return trapezoid(t, t0, depth, t14, tau)
    off = (np.arange(5) + 0.5) / 5 - 0.5
    return np.mean([trapezoid(t + o * cadence_d, t0, depth, t14, tau) for o in off], axis=0)


def _profile_err(resfun, best_x, i, lo, hi, chi2_min, scale, span, x_scale=None):
    """1-sigma error on parameter i: where the delta-chi2 profile (others re-optimized) crosses `scale`.

    Linear interpolation between grid points on each side; capped at span if it never crosses.
    """
    from scipy.optimize import least_squares

    free = [j for j in range(len(best_x)) if j != i]
    lo_f, hi_f = np.array(lo, float)[free], np.array(hi, float)[free]
    xs = None if x_scale is None else np.array(x_scale, float)[free]

    def prof(g):
        def r(q):
            x = np.array(best_x, float)
            x[free] = q
            x[i] = g
            return resfun(x)
        x0 = np.clip(np.array(best_x, float)[free], lo_f + 1e-9, hi_f - 1e-9)
        return 2 * least_squares(r, x0, bounds=(lo_f, hi_f), x_scale=xs if xs is not None else 1.0).cost - chi2_min

    half = []
    for sign in (-1, 1):
        prev_x, prev_y, cross = 0.0, 0.0, span
        for x in span * np.geomspace(0.01, 1, 25):
            g = best_x[i] + sign * x
            if not (lo[i] <= g <= hi[i]):
                cross = prev_x
                break
            y = prof(g)
            if y >= scale:
                cross = prev_x + (x - prev_x) * (scale - prev_y) / max(y - prev_y, 1e-12)
                break
            prev_x, prev_y = x, max(y, 0.0)
        half.append(cross)
    return float(np.mean(half))


def fit_event(t, f, t0, dur_d, depth):
    """Least-squares trapezoid and V fits around one event. Returns fitted values, errors and shape test."""
    from scipy.optimize import least_squares

    w = np.abs(t - t0) <= 2.0 * dur_d
    t, f = t[w], f[w]
    cad = float(np.median(np.diff(t))) if len(t) > 2 else 0.0
    oot = np.abs(t - t0) > 0.75 * dur_d
    sig = robust_std(f[oot]) if oot.sum() > 10 else robust_std(f)
    out = {"n_points": int(len(t)), "noise_ppm": round(sig * 1e6, 1)}
    if len(t) < 20 or not np.isfinite(sig) or sig <= 0:
        return dict(out, status="no_data")

    def res(p, v=False):
        t0_, d_, t14_, tf_ = p
        tau = t14_ / 2 if v else t14_ * tf_
        return (f - binned_model(t, t0_, d_, t14_, tau, cad)) / sig

    lo = [t0 - 0.5 * dur_d, 0, 0.2 * dur_d, 0.01]
    hi = [t0 + 0.5 * dur_d, 1, 3.0 * dur_d, 0.5]
    best = None
    for tf0 in (0.05, 0.15, 0.3):
        r = least_squares(res, [t0, max(depth, 1e-5), dur_d, tf0], bounds=(lo, hi), x_scale=[0.01, 1e-3, 0.01, 0.1])
        if best is None or r.cost < best.cost:
            best = r
    rv = least_squares(lambda p: res([p[0], p[1], p[2], 0.5], v=True), [best.x[0], best.x[1], best.x[2]],
                       bounds=(lo[:3], hi[:3]), x_scale=[0.01, 1e-3, 0.01])
    chi2, chi2_v = 2 * best.cost, 2 * rv.cost
    dof = max(len(t) - 4, 1)
    J = best.jac
    try:
        cov = np.linalg.inv(J.T @ J) * max(1.0, chi2 / dof)
        dg = np.diag(cov)
        err = np.where(dg > 0, np.sqrt(np.abs(dg)), np.nan)
    except np.linalg.LinAlgError:
        err = np.full(4, np.nan)
    # t0 and T14 trade off when an ingress or egress falls in a gap, and the Jacobian then lies:
    # always take their errors from a delta-chi2 profile (others re-optimized)
    scale = max(1.0, chi2 / dof)
    xsc = [0.01, 1e-3, 0.01, 0.1]
    err[0] = _profile_err(res, best.x, 0, lo, hi, chi2, scale, 0.25 * dur_d, xsc)
    err[2] = _profile_err(res, best.x, 2, lo, hi, chi2, scale, 0.5 * dur_d, xsc)
    t0f, df, t14f, tff = best.x
    # ingress or egress inside a data gap: the edge time is only known to within the gap, so the
    # T14 error gets the gap's uniform-distribution width (gap / sqrt 12) in quadrature
    gap_h = 0.0
    big = np.where(np.diff(t) > 2.5 * cad)[0] if cad > 0 else []
    for te in (t0f - t14f / 2, t0f + t14f / 2):
        a, b = te - tff * t14f - cad, te + tff * t14f + cad
        for j in big:
            if t[j] < b and t[j + 1] > a:
                gap_h = max(gap_h, (t[j + 1] - t[j]) * 24)
    if t.min() > t0f - t14f / 2 or t.max() < t0f + t14f / 2:
        gap_h = max(gap_h, t14f * 24)
    out["partial"], out["edge_gap_h"] = bool(gap_h > 0), round(gap_h, 2)
    if gap_h > 0:   # t0 = (ingress + egress) / 2, so it inherits half of that width
        err[2] = math.hypot(err[2], gap_h / 24 / math.sqrt(12))
        err[0] = math.hypot(err[0], gap_h / 24 / math.sqrt(12) / 2)
    dchi2 = chi2_v - chi2
    shape = ("flat_bottom" if dchi2 >= SHAPE_DCHI2 and tff < V_INGRESS_FRAC else
             "v_shaped" if tff >= V_INGRESS_FRAC else "unresolved")
    return dict(out, status="ok", t0_btjd=round(float(t0f), 5), t0_err_d=round(float(err[0]), 5),
                depth_ppm=round(float(df) * 1e6, 1), depth_err_ppm=round(float(err[1]) * 1e6, 1),
                t14_h=round(float(t14f) * 24, 3), t14_err_h=round(float(err[2]) * 24, 3),
                ingress_frac=round(float(tff), 3), chi2=round(chi2, 1), chi2_v=round(chi2_v, 1), dchi2_v_minus_trap=round(dchi2, 1),
                shape=shape, planet_shaped=shape == "flat_bottom", snr=round(float(df) / float(err[1]), 1) if err[1] > 0 else None)


def event_series(secs, tc, dur_d):
    for s in secs:
        if s["time"].size and np.nanmin(s["time"]) - dur_d <= tc <= np.nanmax(s["time"]) + dur_d:
            ok = s["good"] & np.isfinite(s["flat"])
            return s["time"][ok], s["flat"][ok], s
    return None, None, None


# ---------------------------------------------------------------- aliases
def classify_window(ev, ref_ppm, floor=0.10):
    """ref_ppm must come from the same estimator (measure_event box depth at the two events).
    floor: fractional depth systematic (sector-to-sector crowding / detrending) for the 'seen' test."""
    if ev is None or ev["depth_ppm"] is None or ev["flags"]["missing_coverage"]:
        return "unresolved"
    d, e = ev["depth_ppm"], ev["depth_err_ppm"]
    if not e or e <= 0:
        return "unresolved"
    if d / e >= SIGMA and abs(d - ref_ppm) / math.hypot(e, floor * ref_ppm) < SIGMA:
        return "seen"
    if ev["flags"]["one_sided_baseline"] or ev["flags"]["quality_bits_in_window"]:
        return "unresolved"
    if (ref_ppm - d) / e >= SIGMA and d / e < SIGMA:
        return "excluded"
    return "unresolved"


def in_coverage(secs, tc, dur_d):
    for s in secs:
        if np.any(s["good"] & (np.abs(s["time"] - tc) <= 0.5 * dur_d)):
            return s
    return None


def solve_aliases(secs, t1, t1_err, t2, t2_err, dur_d, ref_ppm, pmin):
    dT = t2 - t1
    tmin = min(np.nanmin(s["time"]) for s in secs) - 1
    tmax = max(np.nanmax(s["time"]) for s in secs) + 1
    out = []
    for n in range(1, int(math.floor(dT / pmin)) + 1):
        P = dT / n
        ks = [k for k in range(int(math.floor((tmin - t1) / P)), int(math.ceil((tmax - t1) / P)) + 1) if k not in (0, n)]
        windows = []
        for k in ks:
            tc = t1 + k * P
            sig_t = math.hypot((1 - k / n) * t1_err, (k / n) * t2_err)
            reach = 3 * sig_t
            s = in_coverage(secs, tc, dur_d + 2 * reach)
            if s is None:
                continue
            # timing uncertainty: measure every trial centre within +-3 sigma and keep the most dip-like one,
            # so a transit is never "excluded" just because the prediction was a little off
            trials = tc + np.arange(-reach, reach + 1e-9, max(dur_d / 8, 1e-3)) if reach > dur_d / 16 else [tc]
            evs = [measure_event(s, x, dur_d) for x in trials]
            usable = [e for e in evs if e["depth_ppm"] is not None and not e["flags"]["missing_coverage"]]
            ev = max(usable, key=lambda e: e["snr"] or -1e9) if usable else evs[len(evs) // 2]
            windows.append({"k": k, "t_pred_btjd": round(tc, 4), "t_err_d": round(sig_t, 4), "sector": s["sector"],
                            "t_best_btjd": ev["t_pred_btjd"], "n_trials": len(evs),
                            "depth_ppm": ev["depth_ppm"], "depth_err_ppm": ev["depth_err_ppm"], "snr": ev["snr"],
                            "coverage_frac": ev["coverage_frac"], "event_status": ev["status"],
                            "flags": [f for f, v in ev["flags"].items() if v],
                            "verdict": classify_window(ev, ref_ppm) if usable else "unresolved"})
        verdicts = {w["verdict"] for w in windows}
        status = ("excluded" if "excluded" in verdicts else "seen" if "seen" in verdicts else
                  "unresolved" if "unresolved" in verdicts else "open")
        out.append({"n": n, "period_d": round(P, 5), "period_err_d": round(math.hypot(t1_err, t2_err) / n, 5),
                    "status": status, "n_windows_in_coverage": len(windows),
                    "n_excluded": sum(w["verdict"] == "excluded" for w in windows),
                    "n_seen": sum(w["verdict"] == "seen" for w in windows), "windows": windows})
    return out


def next_windows(t1, t1_err, t2, t2_err, P, n, dur_d, after_btjd, count=3):
    k0 = int(math.ceil((after_btjd - t1) / P))
    rows = []
    for k in range(k0, k0 + count):
        tc = t1 + k * P
        sig = math.hypot((1 - k / n) * t1_err, (k / n) * t2_err)
        if not np.isfinite(sig) or sig > 365:
            rows.append({"k": k, "t_mid_btjd": round(tc, 4), "t_err_d": None, "note": "timing too uncertain"})
            continue
        half = 3 * sig + 0.5 * dur_d
        rows.append({"k": k, "t_mid_btjd": round(tc, 4), "t_err_d": round(sig, 4),
                     "mid_utc": btjd_to_datetime(tc).strftime("%Y-%m-%d %H:%M"),
                     "mid_et": btjd_to_datetime(tc).astimezone(ET).strftime("%Y-%m-%d %H:%M %Z"),
                     "window_start_utc": btjd_to_datetime(tc - half).strftime("%Y-%m-%d %H:%M"),
                     "window_end_utc": btjd_to_datetime(tc + half).strftime("%Y-%m-%d %H:%M")})
    return rows


# ---------------------------------------------------------------- gate + main
def duo_gate(e1, e2, aliases):
    tests = {}
    if e1.get("status") != "ok" or e2.get("status") != "ok":
        tests["events_consistent"] = {"result": "fail", "why": "an event could not be fitted"}
    else:
        dd = abs(e1["depth_ppm"] - e2["depth_ppm"])
        sd = math.sqrt(e1["depth_err_ppm"] ** 2 + e2["depth_err_ppm"] ** 2 +
                       (DEPTH_FLOOR * 0.5 * (e1["depth_ppm"] + e2["depth_ppm"])) ** 2)
        dt = abs(e1["t14_h"] - e2["t14_h"])
        st = math.hypot(e1["t14_err_h"], e2["t14_err_h"])
        why = []
        if dd / sd >= SIGMA:
            why.append(f"depths differ by {dd / sd:.1f} sigma")
        if dt / st >= SIGMA:
            why.append(f"durations differ by {dt / st:.1f} sigma")
        for i, e in ((1, e1), (2, e2)):
            if not e["planet_shaped"]:
                why.append(f"event {i} is {e['shape']} (ingress/T14 {e['ingress_frac']}, dchi2 {e['dchi2_v_minus_trap']})")
        tests["events_consistent"] = {"result": "fail" if why else "pass", "why": why,
                                      "depth_sigma": round(dd / sd, 2), "duration_sigma": round(dt / st, 2)}
    unresolved = [a["period_d"] for a in aliases if a["status"] == "unresolved"]
    tests["coverage_excludes"] = {"result": "fail" if unresolved else "pass",
                                  "why": [f"{len(unresolved)} alias(es) predict a transit in coverage that the data cannot decide"]
                                  if unresolved else [], "unresolved_periods": unresolved}
    seen = sorted([a for a in aliases if a["status"] == "seen"], key=lambda a: (-a["n_seen"], a["n"]))
    # windows at T2 + m*(T2-T1) are shared by every alias, so rank by how many windows each alias explains
    label = ("duo candidate" if not seen else
             f"predicted transit observed: {len(seen)} alias(es) have a third dip in the data; most windows explained by "
             f"P = {seen[0]['period_d']} d ({seen[0]['n_seen']} dips seen, none excluded). No longer a duo: run allsector_gate "
             f"on that period")
    tests["label"] = {"result": "duo" if not seen else "periodic", "label": label,
                      "seen_ranked": [{"period_d": a["period_d"], "n_seen": a["n_seen"]} for a in seen[:10]]}
    verdict = "pass" if all(tests[k]["result"] == "pass" for k in ("events_consistent", "coverage_excludes")) else "fail"
    return verdict, tests, label


def solve(tic, t1, t2, duration_h, depth_ppm, t1_err=0.01, t2_err=0.01, pmin=10.0, rho_solar=None, cache_dir=None,
          lc_files=None, secs=None, refit=True, out_dir=None, after_btjd=None, sectors=None):
    tic = parse_tic(tic)
    t1, t2 = sorted((float(t1), float(t2)))
    dur_d = float(duration_h) / 24
    load = {"source": "given"}
    if secs is None:
        secs, load = load_sectors(tic, cache_dir, lc_files)
    if sectors:
        secs = [s for s in secs if s["sector"] in set(sectors)]
    prepare(secs, (t1, t2), dur_d)
    events = []
    for tc, terr in ((t1, t1_err), (t2, t2_err)):
        t, f, s = event_series(secs, tc, dur_d)
        e = fit_event(t, f, tc, dur_d, depth_ppm * 1e-6) if t is not None else {"status": "no_data"}
        e.update(t_input_btjd=tc, t_input_err_d=terr, sector=s["sector"] if s else None)
        events.append(e)
    if refit and all(e["status"] == "ok" for e in events):
        t1, t2 = events[0]["t0_btjd"], events[1]["t0_btjd"]
        t1_err = max(events[0]["t0_err_d"], 1e-4)
        t2_err = max(events[1]["t0_err_d"], 1e-4)
        dur_d = 0.5 * (events[0]["t14_h"] + events[1]["t14_h"]) / 24
        prepare(secs, (t1, t2), dur_d)
    # reference depth for the alias windows, measured with the same box estimator as the windows
    box = []
    for tc in (t1, t2):
        _, _, s = event_series(secs, tc, dur_d)
        if s is not None:
            ev = measure_event(s, tc, dur_d)
            if ev["depth_ppm"] is not None:
                box.append(ev["depth_ppm"])
    ref = float(np.mean(box)) if box else float(depth_ppm)
    aliases = solve_aliases(secs, t1, t1_err, t2, t2_err, dur_d, ref, pmin)
    after = after_btjd if after_btjd is not None else now_btjd()
    from .stellar_density import expected_duration_h

    fit_depth = [e["depth_ppm"] for e in events if e.get("status") == "ok"]
    k = math.sqrt(max(np.mean(fit_depth) if fit_depth else ref, 0) * 1e-6)
    for a in aliases:
        if a["status"] != "excluded":
            a["next_windows"] = next_windows(t1, t1_err, t2, t2_err, a["period_d"], a["n"], dur_d, after)
        if rho_solar:
            a["expected_t14_h_b0"] = round(expected_duration_h(a["period_d"], rho_solar, k), 2)
            a["t14_ratio_obs_over_b0"] = round(dur_d * 24 / a["expected_t14_h_b0"], 2)
    verdict, tests, label = duo_gate(events[0], events[1], aliases)
    surv = [a for a in aliases if a["status"] != "excluded"]
    result = {"tool": "duo_solver", "tic": tic, "generated_at": now_et(),
              "inputs": {"t1": t1, "t1_err_d": t1_err, "t2": t2, "t2_err_d": t2_err, "duration_h": round(dur_d * 24, 3),
                         "ref_box_depth_ppm": round(ref, 1), "pmin_d": pmin, "rho_solar": rho_solar, "refit": refit},
              "sectors": [s["sector"] for s in secs], "events": events, "delta_t_d": round(t2 - t1, 5),
              "n_aliases": len(aliases), "n_excluded": len(aliases) - len(surv), "surviving_periods": [a["period_d"] for a in surv],
              "aliases": aliases, "gate": {"verdict": verdict, "tests": tests}, "label": label, "load": load,
              "note": "A duo candidate is not a planet. Surviving aliases are hypotheses to schedule follow-up for."}
    out_dir = Path(out_dir or Path(cache_dir or default_cache_dir()) / "duo_solver" / f"TIC{tic}_{t1:.2f}_{t2:.2f}")
    write_json(out_dir / "duo.json", result)
    with (out_dir / "aliases.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "period_d", "period_err_d", "status", "n_windows_in_coverage", "n_excluded", "n_seen",
                    "expected_t14_h_b0", "next_mid_utc"])
        for a in aliases:
            w.writerow([a["n"], a["period_d"], a["period_err_d"], a["status"], a["n_windows_in_coverage"], a["n_excluded"],
                        a["n_seen"], a.get("expected_t14_h_b0"), (a.get("next_windows") or [{}])[0].get("mid_utc")])
    result["json"], result["csv"] = str(out_dir / "duo.json"), str(out_dir / "aliases.csv")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tic")
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--t2", type=float, required=True)
    ap.add_argument("--t1-err", type=float, default=0.01, help="days")
    ap.add_argument("--t2-err", type=float, default=0.01)
    ap.add_argument("--duration", type=float, required=True, help="hours")
    ap.add_argument("--depth", type=float, required=True, help="ppm")
    ap.add_argument("--pmin", type=float, default=10.0)
    ap.add_argument("--rho", type=float, help="stellar density, solar units")
    ap.add_argument("--mstar", type=float)
    ap.add_argument("--rstar", type=float)
    ap.add_argument("--no-refit", action="store_true", help="use the given epochs, depth and duration as-is")
    ap.add_argument("--lc-file", action="append")
    ap.add_argument("--sectors", type=lambda s: [int(x) for x in s.split(",") if x], help="only use these sectors")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args(argv)
    rho = a.rho or (a.mstar / a.rstar ** 3 if a.mstar and a.rstar else None)
    r = solve(a.tic, a.t1, a.t2, a.duration, a.depth, a.t1_err, a.t2_err, a.pmin, rho, a.cache_dir, a.lc_file,
              refit=not a.no_refit, out_dir=a.out_dir, sectors=a.sectors)
    print(json.dumps(jsonable({k: r[k] for k in ("tic", "inputs", "delta_t_d", "n_aliases", "n_excluded", "surviving_periods",
                                                 "label", "json", "csv")} |
                              {"gate": r["gate"]["verdict"], "gate_tests": r["gate"]["tests"],
                               "events": [{k: e.get(k) for k in ("sector", "t0_btjd", "t0_err_d", "depth_ppm", "depth_err_ppm",
                                                                 "t14_h", "t14_err_h", "shape")} for e in r["events"]],
                               "surviving": [{k: x.get(k) for k in ("n", "period_d", "status", "n_windows_in_coverage",
                                                                    "expected_t14_h_b0")} | {"next": (x.get("next_windows") or [{}])[0].get("mid_utc")}
                                             for x in r["aliases"] if x["status"] != "excluded"]}), indent=2))
    return 0 if r["gate"]["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
