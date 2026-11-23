"""allsector_gate: does a trial transit ephemeris hold up in every TESS sector?

Given a TIC and a trial (period, epoch, duration), this measures the local depth at
every predicted transit in every available sector, then runs hard tests:

  orbit_harmonic   period within tol (5%) of 6.85 / 13.7 / 27.4 d          -> fail
  missing_transit  a clean, well-covered predicted transit shows no dip      -> fail
                   (depth >= 3 sigma below the reference AND < 3 sigma above 0)
  repeats          detected (>= 3 sigma) in >= 2 sectors, or, with
                   claimed_sectors, in >= 1 sector outside them              -> fail / incomplete
  odd_even         |odd - even| / sigma > 3 (fails closed if a parity is
                   unobserved)                                               -> fail / incomplete
  depth_consistency one clean event > 5 sigma from the mean of the others
                   (sigma includes a 10%-of-depth systematic floor)          -> fail
  secondary        phase-0.5 dip with SNR >= 5 and >= 10% of primary depth   -> fail

Verdict: fail if any test fails; else incomplete if any test could not run; else pass.
"pass" means the ephemeris survived these checks. It does not make a planet.

    allsector-gate 261136679 --period 6.2678139 --epoch 3901.578202 --duration 2.789
    allsector-gate 288183107 --period 12.96393 --epoch 3319.20707 --duration 6.208 --depth 2140 --claimed-sectors 74,75
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from .common import default_cache_dir, jsonable, now_et, parse_tic, write_json
from .harmonics import period_harmonic_flag

# TESS quality bits (lightkurve.utils.TessQualityFlags)
DESAT = 32                   # reaction-wheel desaturation = momentum dump
STRAY = 2048 | 4096          # Straylight, Straylight2
DEFAULT_BAD = 17087          # lightkurve "default" bitmask: cadences we drop
ODD_EVEN_SIGMA = 3.0
MISSING_SIGMA = 3.0
DETECT_SIGMA = 3.0
CONSISTENCY_SIGMA = 5.0
CONSISTENCY_FLOOR = 0.10     # fraction of reference depth added in quadrature
SECONDARY_SNR = 5.0
SECONDARY_RATIO = 0.10
GAP_D = 0.5                  # a gap longer than this splits a sector into segments


# ---------------------------------------------------------------- loading
def read_lc_file(path: Path) -> dict:
    """One sector from a SPOC / TESS-SPOC / QLP FITS file, or from a fixture CSV(.gz)."""
    path = Path(path)
    if path.suffix in (".csv", ".gz"):
        import pandas as pd

        df = pd.read_csv(path)
        out = []
        for sec, g in df.groupby("sector"):
            out.append({"sector": int(sec), "author": str(g["author"].iloc[0]) if "author" in g else "fixture",
                        "cadence_s": float(g["cadence_s"].iloc[0]) if "cadence_s" in g else np.nan,
                        "time": g["time"].to_numpy(float), "flux": g["flux"].to_numpy(float),
                        "flux_err": g["flux_err"].to_numpy(float), "quality": g["quality"].to_numpy(int),
                        "source": str(path)})
        return out
    from astropy.io import fits

    with fits.open(path) as h:
        hdr0, hdr, d = h[0].header, h[1].header, h[1].data
        names = d.columns.names
        fcol = next(c for c in ("PDCSAP_FLUX", "KSPSAP_FLUX", "SAP_FLUX") if c in names)
        ecol = fcol + "_ERR" if fcol + "_ERR" in names else None
        author = "QLP" if "QLP" in fcol or "qlp" in path.name else ("TESS-SPOC" if "tess-spoc" in path.name else "SPOC")
        t = np.asarray(d["TIME"], float)
        return [{"sector": int(hdr0.get("SECTOR", -1)), "author": author,
                 "cadence_s": float(np.nanmedian(np.diff(t)) * 86400) if len(t) > 2 else np.nan,
                 "time": t, "flux": np.asarray(d[fcol], float),
                 "flux_err": np.asarray(d[ecol], float) if ecol else np.full(len(t), np.nan),
                 "quality": np.asarray(d["QUALITY"], int) if "QUALITY" in names else np.zeros(len(t), int),
                 "crowdsap": hdr.get("CROWDSAP"), "source": str(path)}]


def load_sectors(tic: int, cache_dir=None, lc_files=None) -> tuple[list[dict], dict]:
    """Sectors from explicit files, else one best product per sector via tess_lc_fetch."""
    info = {}
    if lc_files:
        files = list(lc_files)
        info["source"] = "lc_files"
    else:
        from .tess_lc_fetch import fetch

        _, m = fetch(tic, cache_dir=cache_dir, pick="best", load=False)
        files = [p["local_path"] for p in m["products"]]
        info = {"source": "tess_lc_fetch", "manifest_hash": m["manifest_hash"],
                "manifest": m["manifest_paths"][0], "fetch_failures": m["failures"]}
    secs = [s for f in files for s in read_lc_file(f)]
    secs.sort(key=lambda s: s["sector"])
    return secs, info


# ---------------------------------------------------------------- detrending
def detrend(t, f, good, in_transit, window_d):
    """Biweight trend fitted with predicted transits masked, interpolated across them."""
    from wotan import flatten

    fit = good & ~in_transit & np.isfinite(f)
    flat = np.full_like(f, np.nan)
    if fit.sum() < 20:
        return flat
    med = np.nanmedian(f[fit])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trend_fit = flatten(t[fit], f[fit] / med, window_length=window_d, method="biweight",
                            return_trend=True, break_tolerance=GAP_D)[1]
    ok = np.isfinite(trend_fit)
    trend = np.interp(t, t[fit][ok], trend_fit[ok])
    flat = f / med / trend
    return flat


def segments(t):
    """(start, end) of contiguous data stretches (split at gaps > GAP_D)."""
    t = np.sort(t[np.isfinite(t)])
    if len(t) == 0:
        return []
    cut = np.where(np.diff(t) > GAP_D)[0]
    starts = np.r_[t[0], t[cut + 1]]
    ends = np.r_[t[cut], t[-1]]
    return list(zip(starts, ends))


from .folding import robust_std  # noqa: E402  (re-exported for transit_search / secondary_dilution)


# ---------------------------------------------------------------- per-event measurement
def measure_event(sec: dict, tc: float, dur_d: float) -> dict:
    t, flat, q, good = sec["time"], sec["flat"], sec["quality"], sec["good"]
    dt = t - tc
    core = np.abs(dt) <= 0.4 * dur_d          # central 80% of the transit
    oot = (np.abs(dt) > 1.0 * dur_d) & (np.abs(dt) < 3.0 * dur_d)
    win = np.abs(dt) <= 1.0 * dur_d
    cin, cout = core & good & np.isfinite(flat), oot & good & np.isfinite(flat)
    cad_d = np.nanmedian(np.diff(t)) if len(t) > 2 else np.nan
    expected = max(1, int(round(0.8 * dur_d / cad_d))) if np.isfinite(cad_d) and cad_d > 0 else 1
    n_pre, n_post = int((cout & (dt < 0)).sum()), int((cout & (dt > 0)).sum())
    bits = int(np.bitwise_or.reduce(q[win])) if win.any() else 0
    frac_bad = float(((q[win] & DEFAULT_BAD) != 0).mean()) if win.any() else 0.0
    seg = segments(t[np.isfinite(sec["flux"])])
    edge_h = min((min(abs(tc - a), abs(b - tc)) * 24 for a, b in seg if a - dur_d <= tc <= b + dur_d), default=None)
    ev = {
        "sector": sec["sector"], "author": sec["author"], "t_pred_btjd": round(float(tc), 5),
        "n_in": int(cin.sum()), "n_in_any": int(core.sum()), "n_in_expected": expected,
        "n_oot": int(cout.sum()), "n_pre": n_pre, "n_post": n_post,
        "coverage_frac": round(cin.sum() / expected, 3),
        "quality_bits": bits, "quality_names": quality_names(bits),
        "frac_window_flagged": round(frac_bad, 3),
        "flags": {
            "missing_coverage": bool(cin.sum() < max(3, 0.5 * expected)),
            # isolated flagged cadences are already dropped; flag the event only if >10% of the window is bad
            "quality_bits_in_window": frac_bad > 0.10,
            "stray_light": bool(bits & STRAY),
            "momentum_dump": bool(bits & DESAT),
            "near_edge": bool(edge_h is None or edge_h < 24 * dur_d),
            "one_sided_baseline": bool(n_pre < 3 or n_post < 3),
        },
        "hours_to_segment_edge": None if edge_h is None else round(edge_h, 2),
        "depth_ppm": None, "depth_err_ppm": None, "snr": None,
    }
    if cin.sum() >= 3 and cout.sum() >= 6:
        base = np.nanmedian(flat[cout])
        depth = base - np.nanmean(flat[cin])
        s = robust_std(flat[cout])
        err = s * np.sqrt(1.0 / cin.sum() + 1.0 / cout.sum())
        ev.update(depth_ppm=round(depth * 1e6, 1), depth_err_ppm=round(err * 1e6, 1),
                  snr=round(depth / err, 2) if err > 0 else None)
    if ev["depth_ppm"] is None or ev["flags"]["missing_coverage"]:
        ev["status"] = "no_coverage"
    elif any(ev["flags"][k] for k in ("quality_bits_in_window", "stray_light", "momentum_dump", "near_edge",
                                      "one_sided_baseline")):
        ev["status"] = "flagged"
    else:
        ev["status"] = "clean"
    return ev


QUALITY_NAMES = {1: "AttitudeTweak", 2: "SafeMode", 4: "CoarsePoint", 8: "EarthPoint", 16: "Argabrightening",
                 32: "Desat", 64: "ApertureCosmic", 128: "ManualExclude", 256: "Discontinuity",
                 512: "ImpulsiveOutlier", 1024: "CollateralCosmic", 2048: "Straylight", 4096: "Straylight2",
                 8192: "PlanetSearchExclude", 16384: "BadCalibrationExclude", 32768: "InsufficientTargets"}


def quality_names(bits: int) -> list[str]:
    return [n for b, n in QUALITY_NAMES.items() if bits & b]


def wmean(d, e):
    d, e = np.asarray(d, float), np.asarray(e, float)
    if len(d) == 0:
        return None, None
    w = 1 / e ** 2
    return float((w * d).sum() / w.sum()), float(1 / np.sqrt(w.sum()))


def error_scale(evs) -> float:
    """Robust excess-scatter factor for per-event depth errors, floored at 1.

    1.4826 * MAD of the normalized residuals about the weighted mean. Using the MAD
    (not chi2) keeps a single bad event from inflating everyone's errors and hiding itself.
    """
    if len(evs) < 5:
        return 1.0
    d = np.array([e["depth_ppm"] for e in evs])
    e = np.array([e["depth_err_ppm"] for e in evs])
    m, _ = wmean(d, e)
    z = (d - m) / e
    return float(max(1.0, 1.4826 * np.median(np.abs(z - np.median(z)))))


def fold_depth(secs, t0, period, dur_d, phase_center):
    """Robust folded depth at a phase (see folding.phase_depth)."""
    from .folding import phase_depth

    if not secs:
        return {"depth_ppm": None, "err_ppm": None, "snr": None, "n_in": 0}
    t = np.concatenate([s["time"][s["good"]] for s in secs])
    f = np.concatenate([s["flat"][s["good"]] for s in secs])
    ok = np.isfinite(f)
    return phase_depth(t[ok], f[ok], t0, period, dur_d, phase_center, exclude_primary=phase_center != 0.0)


def prepare_sectors(secs, period_d, epoch_btjd, dur_d, window_d=None):
    """Add 'good' (quality mask), 'flat' (detrended, transits masked) and 'epochs' to each sector dict.

    With period_d=None nothing is masked (used by transit_search before any signal is known).
    """
    for s in secs:
        t, f, q = s["time"], s["flux"], s["quality"]
        fin = np.isfinite(t) & np.isfinite(f)
        bad_bits = DEFAULT_BAD if s["author"] != "QLP" else ~0  # QLP: any nonzero flag is bad
        s["good"] = fin & ((q & bad_bits) == 0)
        in_tr = np.zeros(len(t), bool)
        n = np.array([], int)
        if period_d and fin.any():
            n = np.arange(np.floor((np.nanmin(t[fin]) - epoch_btjd) / period_d),
                          np.ceil((np.nanmax(t[fin]) - epoch_btjd) / period_d) + 1).astype(int)
            for tc in epoch_btjd + n * period_d:
                in_tr |= np.abs(t - tc) <= 0.75 * dur_d
        s["epochs"] = n
        s["flat"] = detrend(t, f, s["good"], in_tr, window_d or max(0.5, 3.0 * dur_d))
    return secs


# ---------------------------------------------------------------- gate
def gate(tic, period_d, epoch_btjd, duration_h, depth_ppm=None, claimed_sectors=None,
         cache_dir=None, lc_files=None, out_dir=None, harmonic_tol=0.05, plot=True, rho_solar=None) -> dict:
    tic = parse_tic(tic)
    P, T0, dur_d = float(period_d), float(epoch_btjd), float(duration_h) / 24.0
    claimed = sorted({int(s) for s in claimed_sectors}) if claimed_sectors else None
    secs, load_info = load_sectors(tic, cache_dir, lc_files)

    prepare_sectors(secs, P, T0, dur_d)
    events = []
    for s in secs:
        t, fin = s["time"], np.isfinite(s["time"]) & np.isfinite(s["flux"])
        for k, tc in zip(s["epochs"], T0 + s["epochs"] * P):
            if np.nanmin(t[fin]) - dur_d <= tc <= np.nanmax(t[fin]) + dur_d:
                ev = measure_event(s, tc, dur_d)
                ev["epoch"] = int(k)
                events.append(ev)

    clean = [e for e in events if e["status"] == "clean"]
    tests, reasons = {}, []

    # reference depth
    if depth_ppm:
        ref, ref_src = float(depth_ppm), "given"
    else:
        pool = [e for e in clean if (claimed is None or e["sector"] in claimed)]
        ref = wmean([e["depth_ppm"] for e in pool], [e["depth_err_ppm"] for e in pool])[0] if pool else None
        ref_src = "claimed_sectors_mean" if claimed else "all_clean_mean"

    # 1. harmonic
    h = period_harmonic_flag(P, harmonic_tol)
    tests["orbit_harmonic"] = {"result": "fail" if h["flagged"] else "pass", **h}

    # 2. missing transits
    for e in events:
        e["missing"] = None
        e["detected"] = None
        if e["status"] == "clean":
            e["detected"] = e["snr"] is not None and e["snr"] >= DETECT_SIGMA
            if ref is not None:
                e["missing"] = bool((ref - e["depth_ppm"]) / e["depth_err_ppm"] >= MISSING_SIGMA
                                    and e["depth_ppm"] / e["depth_err_ppm"] < DETECT_SIGMA)
    missing = [e for e in clean if e["missing"]]
    if ref is None or len(clean) < 2:
        tests["missing_transit"] = {"result": "incomplete", "reason": "fewer than 2 clean events or no reference depth",
                                    "n_clean": len(clean)}
    else:
        tests["missing_transit"] = {
            "result": "fail" if missing else "pass", "threshold_sigma": MISSING_SIGMA, "n_clean": len(clean),
            "n_missing": len(missing),
            "missing": [{k: e[k] for k in ("epoch", "sector", "t_pred_btjd", "depth_ppm", "depth_err_ppm")} for e in missing]}

    # 3. repeats in independent sectors
    det_secs = sorted({e["sector"] for e in clean if e["detected"] and not e["missing"]})
    clean_secs = sorted({e["sector"] for e in clean})
    if claimed:
        indep_det = [s for s in det_secs if s not in claimed]
        indep_clean = [s for s in clean_secs if s not in claimed]
        ok = bool(indep_det)
        rule = "detected in >= 1 sector outside claimed_sectors"
    else:
        indep_det, indep_clean = det_secs, clean_secs
        ok = len(det_secs) >= 2
        rule = "detected in >= 2 distinct sectors"
    if ok:
        r = "pass"
    elif (claimed and indep_clean) or (not claimed and len(clean_secs) >= 2):
        r = "fail"
    else:
        r = "incomplete"
    tests["repeats"] = {"result": r, "rule": rule, "repeating": ok, "detected_sectors": det_secs,
                        "clean_sectors": clean_secs, "independent_detected_sectors": indep_det,
                        "independent_clean_sectors": indep_clean}

    # 4. odd / even (hard gate, fails closed)
    odd = [e for e in clean if e["epoch"] % 2]
    even = [e for e in clean if not e["epoch"] % 2]
    if not odd or not even:
        tests["odd_even"] = {"result": "incomplete", "threshold_sigma": ODD_EVEN_SIGMA,
                             "reason": f"clean odd={len(odd)} even={len(even)}; need both"}
    else:
        # White-noise errors understate real per-event scatter (red noise, detrending). Scale them by
        # sqrt(reduced chi2) of all detected clean depths about their mean, never below 1.
        scale = error_scale([e for e in clean if e["detected"]])
        do, eo = wmean([e["depth_ppm"] for e in odd], [e["depth_err_ppm"] * scale for e in odd])
        de, ee = wmean([e["depth_ppm"] for e in even], [e["depth_err_ppm"] * scale for e in even])
        sig = abs(do - de) / np.hypot(eo, ee)
        tests["odd_even"] = {"result": "fail" if sig > ODD_EVEN_SIGMA else "pass", "threshold_sigma": ODD_EVEN_SIGMA,
                             "odd_depth_ppm": round(do, 1), "odd_err_ppm": round(eo, 1), "n_odd": len(odd),
                             "even_depth_ppm": round(de, 1), "even_err_ppm": round(ee, 1), "n_even": len(even),
                             "diff_sigma": round(float(sig), 2), "error_scale": round(scale, 3)}

    # 5. depth consistency among detected clean events
    det = [e for e in clean if e["detected"]]
    if len(det) < 3:  # informational when too few: missing_transit/repeats already cover this case
        tests["depth_consistency"] = {"result": "incomplete", "reason": f"{len(det)} detected clean events; need 3"}
    else:
        scale = error_scale(det)
        worst = None
        for i, e in enumerate(det):
            others = det[:i] + det[i + 1:]
            m, me = wmean([o["depth_ppm"] for o in others], [o["depth_err_ppm"] * scale for o in others])
            z = abs(e["depth_ppm"] - m) / np.sqrt((e["depth_err_ppm"] * scale) ** 2 + me ** 2 + (CONSISTENCY_FLOOR * m) ** 2)
            if worst is None or z > worst[0]:
                worst = (float(z), e)
        tests["depth_consistency"] = {"result": "fail" if worst[0] > CONSISTENCY_SIGMA else "pass",
                                      "threshold_sigma": CONSISTENCY_SIGMA, "systematic_floor_frac": CONSISTENCY_FLOOR,
                                      "max_z": round(worst[0], 2), "worst_epoch": worst[1]["epoch"],
                                      "worst_sector": worst[1]["sector"], "error_scale": round(scale, 3)}

    # 6. secondary eclipse at phase 0.5 (+ scan for eccentric orbits, info only)
    usable = [s for s in secs if np.isfinite(s["flat"]).any()]
    prim = fold_depth(usable, T0, P, dur_d, 0.0)
    sec = fold_depth(usable, T0, P, dur_d, 0.5)
    scan = [fold_depth(usable, T0, P, dur_d, c) for c in np.arange(0.1, 0.91, 0.05)]
    scan = [x for x in scan if x["snr"] is not None]
    pd_ = prim["depth_ppm"] or ref
    if sec["snr"] is None:
        tests["secondary"] = {"result": "incomplete", "reason": "too few points near phase 0.5", "phase_0p5": sec}
    else:
        is_fail = sec["snr"] >= SECONDARY_SNR and pd_ and sec["depth_ppm"] >= SECONDARY_RATIO * pd_
        tests["secondary"] = {"result": "fail" if is_fail else "pass", "snr_threshold": SECONDARY_SNR,
                              "ratio_threshold": SECONDARY_RATIO, "phase_0p5": sec, "primary_fold": prim,
                              "strongest_other_phase": max(scan, key=lambda x: x["snr"]) if scan else None}

    # 7. duration vs stellar density (only when a density is given)
    if rho_solar is not None:
        from .duration_check import check as duration_check

        dep = (prim.get("depth_ppm") if prim.get("depth_ppm") and prim["depth_ppm"] > 0 else None) or ref or 0.0
        tests["duration_density"] = duration_check(P, float(duration_h), dep, rho_solar)

    fails = [k for k, v in tests.items() if v["result"] == "fail"]
    inc = [k for k, v in tests.items() if v["result"] == "incomplete"]
    if not secs:
        verdict, reasons = "incomplete", ["no light curves loaded"]
    elif fails:
        verdict = "fail"
    elif inc:
        verdict = "incomplete"
    else:
        verdict = "pass"
    for k in fails:
        t = tests[k]
        reasons.append({
            "orbit_harmonic": lambda: f"period within {harmonic_tol:.0%} of TESS orbit harmonic {t['near']}",
            "missing_transit": lambda: f"{t['n_missing']} clean predicted transit(s) missing: " + ", ".join(
                f"S{m['sector']} E{m['epoch']} {m['depth_ppm']:.0f}+/-{m['depth_err_ppm']:.0f} ppm" for m in t["missing"]),
            "repeats": lambda: f"does not repeat ({t['rule']}); detected only in sectors {t['detected_sectors']}",
            "odd_even": lambda: f"odd/even differ by {t['diff_sigma']} sigma",
            "depth_consistency": lambda: f"S{t['worst_sector']} E{t['worst_epoch']} depth {t['max_z']} sigma from the rest",
            "secondary": lambda: f"phase-0.5 dip {t['phase_0p5']['depth_ppm']} ppm at SNR {t['phase_0p5']['snr']}",
            "duration_density": lambda: f"duration {t['flag']}: {t['reason']}",
        }[k]())
    for k in inc:
        reasons.append(f"{k} incomplete: {tests[k].get('reason', 'insufficient data')}")

    out_dir = Path(out_dir or (Path(cache_dir or default_cache_dir()) / "allsector_gate" / f"TIC{tic}_P{P:.5f}"))
    result = {
        "tool": "allsector_gate", "tic": tic, "generated_at": now_et(),
        "inputs": {"period_d": P, "epoch_btjd": T0, "duration_h": duration_h, "depth_ppm": depth_ppm,
                   "claimed_sectors": claimed, "rho_solar": rho_solar},
        "verdict": verdict, "reasons": reasons, "reference_depth_ppm": None if ref is None else round(ref, 1),
        "reference_depth_source": ref_src, "tests": tests,
        "sectors": [{"sector": s["sector"], "author": s["author"], "cadence_s": round(s["cadence_s"], 1),
                     "n_good": int(s["good"].sum()), "crowdsap": s.get("crowdsap"), "source": s["source"]} for s in secs],
        "n_events": len(events), "n_clean": len(clean),
        "events": events, "load": load_info,
        "note": "Diagnostics for a known/trial ephemeris. A pass is not a detection or a discovery.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    if plot and secs:
        result["plot"] = str(make_plot(result, usable, out_dir / "gate.png"))
    write_json(out_dir / "gate.json", result)
    result["json"] = str(out_dir / "gate.json")
    return result


def brief(r) -> str:
    """One line: verdict, per-test results, first reason, json path. For agents that pay per token."""
    tt = " ".join(f"{k}={v['result'][0]}" for k, v in r["tests"].items())
    why = r["reasons"][0] if r["reasons"] else ""
    why = why if isinstance(why, str) else str(why)
    return f"TIC{r['tic']} P={r['inputs']['period_d']:.5f} gate={r['verdict']} clean={r['n_clean']}/{r['n_events']} [{tt}] {why[:160]} -> {r['json']}"


def make_plot(res, secs, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P, T0, dur_d = res["inputs"]["period_d"], res["inputs"]["epoch_btjd"], res["inputs"]["duration_h"] / 24
    fig, ax = plt.subplots(3, 1, figsize=(9, 9))
    ev = [e for e in res["events"] if e["depth_ppm"] is not None]
    col = {"clean": "k", "flagged": "orange", "no_coverage": "lightgray"}
    for e in ev:
        c = "red" if e.get("missing") else col[e["status"]]
        ax[0].errorbar(e["t_pred_btjd"], e["depth_ppm"], e["depth_err_ppm"], fmt="o" if e["status"] == "clean" else "s",
                       color=c, ms=4, capsize=2)
    if res["reference_depth_ppm"] is not None:
        ax[0].axhline(res["reference_depth_ppm"], color="C0", ls="--", label=f"reference {res['reference_depth_ppm']:.0f} ppm")
    ax[0].axhline(0, color="gray", lw=0.5)
    for s in sorted({e["sector"] for e in ev}):
        ts = [e["t_pred_btjd"] for e in ev if e["sector"] == s]
        ax[0].annotate(f"S{s}", (np.mean(ts), 0), fontsize=7, ha="center", va="top", color="gray")
    ax[0].set(xlabel="predicted mid-transit [BTJD]", ylabel="depth [ppm]",
              title=f"TIC {res['tic']}  P={P:.5f} d  verdict: {res['verdict'].upper()}  "
                    "(black clean, orange flagged, red missing)")
    ax[0].legend(fontsize=7)
    t = np.concatenate([s["time"][s["good"]] for s in secs])
    f = np.concatenate([s["flat"][s["good"]] for s in secs])
    n = np.round((t - T0) / P).astype(int)
    for k, (center, a) in enumerate(((0.0, ax[1]), (0.5, ax[2]))):
        x = ((((t - T0) / P - center + 0.5) % 1) - 0.5) * P * 24
        win = np.abs(x) < 3 * dur_d * 24
        for par, c in ((0, "C0"), (1, "C3")):
            sel = win & (n % 2 == par) & np.isfinite(f)
            if sel.sum() < 10:
                continue
            bins = np.linspace(-3 * dur_d * 24, 3 * dur_d * 24, 61)
            idx = np.digitize(x[sel], bins)
            xb = [x[sel][idx == i].mean() for i in range(1, len(bins)) if (idx == i).sum() > 2]
            yb = [(f[sel][idx == i].mean() - 1) * 1e6 for i in range(1, len(bins)) if (idx == i).sum() > 2]
            label = ("even" if par == 0 else "odd") if k == 0 else None
            a.plot(xb, yb, ".", color=c, label=label)
        a.axvspan(-dur_d * 12, dur_d * 12, color="gray", alpha=0.15)
        a.set(xlabel=f"hours from phase {center}", ylabel="flux - 1 [ppm]")
    tt = res["tests"]
    ax[1].set_title(f"primary fold, odd vs even: {tt['odd_even']['result']}  " +
                    (f"({tt['odd_even'].get('diff_sigma')} sigma)" if "diff_sigma" in tt["odd_even"] else ""), fontsize=9)
    ax[1].legend(fontsize=7)
    ax[2].set_title(f"secondary (phase 0.5): {tt['secondary']['result']}", fontsize=9)
    txt = "  ".join(f"{k}:{v['result']}" for k, v in tt.items())
    fig.text(0.01, 0.005, txt, fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tic")
    ap.add_argument("--period", type=float, required=True, help="days")
    ap.add_argument("--epoch", type=float, required=True, help="mid-transit, BTJD (BJD - 2457000)")
    ap.add_argument("--duration", type=float, required=True, help="hours")
    ap.add_argument("--depth", type=float, help="claimed depth in ppm (reference for missing-transit test)")
    ap.add_argument("--claimed-sectors", type=lambda s: [int(x) for x in s.split(",") if x])
    ap.add_argument("--lc-file", action="append", help="use these FITS/CSV files instead of tess_lc_fetch")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--harmonic-tol", type=float, default=0.05)
    ap.add_argument("--rho", type=float, help="stellar density (solar); adds the duration_density test")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--brief", action="store_true", help="print one line instead of the JSON summary")
    a = ap.parse_args(argv)
    r = gate(a.tic, a.period, a.epoch, a.duration, a.depth, a.claimed_sectors, a.cache_dir, a.lc_file, a.out_dir,
             a.harmonic_tol, plot=not a.no_plot, rho_solar=a.rho)
    if a.brief:
        print(brief(r))
        return 0
    print(json.dumps(jsonable({"tic": r["tic"], "verdict": r["verdict"], "reasons": r["reasons"],
                               "reference_depth_ppm": r["reference_depth_ppm"], "n_events": r["n_events"],
                               "n_clean": r["n_clean"], "sectors": [s["sector"] for s in r["sectors"]],
                               "tests": {k: {kk: vv for kk, vv in v.items() if kk not in ("refs", "scan")}
                                         for k, v in r["tests"].items()},
                               "json": r["json"], "plot": r.get("plot")}), indent=2))
    return {"pass": 0, "fail": 1, "incomplete": 3}[r["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
