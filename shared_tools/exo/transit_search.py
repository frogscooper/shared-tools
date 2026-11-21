"""transit_search: detrend + BLS periodic search + single-transit dip finder for one TIC.

1. Light curves: one product per sector via tess_lc_fetch (or --lc-file), lightkurve default
   quality mask, wotan biweight detrend (window --window, default 1 d), binned to --bin-min.
2. Periodic: astropy BoxLeastSquares on each observing season (sectors closer than 100 d) with a
   log-period grid, then the best peaks are refined phase-coherently on the full baseline.
   After each signal its transits are masked and the search repeats (--n-signals).
3. Single transits: a sliding box (2-16 h) on the detrended flux. SNR is normalized by the
   robust scatter of all trial boxes of that width, which absorbs red noise. Dips that line
   up with a periodic signal are labeled, not dropped.

Every signal comes out with status "unpromoted". Nothing here is a candidate until
allsector_gate (and catalog_veto) has been run on it.

    transit-search 261136679 --sectors 1,4,8,11,12,13 --pmin 1 --pmax 30
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np

from .allsector_gate import load_sectors, prepare_sectors, robust_std
from .common import default_cache_dir, jsonable, now_et, parse_tic, write_json
from .harmonics import period_harmonic_flag

DURATIONS_H = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0)
SINGLE_DURATIONS_H = (2.0, 4.0, 6.0, 8.0, 12.0, 16.0)
SEASON_GAP_D = 100.0
CSV_COLS = ["rank", "kind", "status", "period_d", "epoch_btjd", "depth_ppm", "depth_err_ppm", "duration_h", "snr", "sde",
            "n_transits", "n_seasons", "odd_even_sigma", "harmonic_flag", "tess_orbit_flag", "harmonic_of"]


# ---------------------------------------------------------------- data
def binned_flat(secs, bin_min):
    """Concatenate detrended good points and bin each sector to bin_min minutes."""
    ts, fs, es, ss = [], [], [], []
    for s in secs:
        ok = s["good"] & np.isfinite(s["flat"])
        t, f = s["time"][ok], s["flat"][ok]
        if len(t) < 10:
            continue
        if np.median(np.diff(t)) * 1440 >= bin_min / 2:
            # FFI cadence (10 or 30 min) already at the bin size: binning would leave one point per bin
            ts.append(t), fs.append(f), es.append(np.full(len(t), robust_std(f))), ss.append(np.full(len(t), s["sector"]))
            continue
        idx = np.floor((t - t[0]) / (bin_min / 1440.0)).astype(int)
        cnt = np.bincount(idx)
        keep = cnt >= 2
        tb = (np.bincount(idx, t)[keep] / cnt[keep])
        fb = (np.bincount(idx, f)[keep] / cnt[keep])
        sig = robust_std(f) / np.sqrt(cnt[keep])
        ts.append(tb), fs.append(fb), es.append(sig), ss.append(np.full(len(tb), s["sector"]))
    return np.concatenate(ts), np.concatenate(fs), np.concatenate(es), np.concatenate(ss)


def seasons(t, sec):
    order = np.argsort(t)
    t, sec = t[order], sec[order]
    cuts = np.where(np.diff(t) > SEASON_GAP_D)[0]
    bounds = np.r_[0, cuts + 1, len(t)]
    return [(t[a], t[b - 1]) for a, b in zip(bounds[:-1], bounds[1:])]


def period_grid(pmin, pmax, baseline, dur_min_d, oversample=3):
    """Geometric grid: dP/P = dur_min / (oversample * baseline)."""
    r = 1 + dur_min_d / (oversample * baseline)
    n = int(np.ceil(np.log(pmax / pmin) / np.log(r)))
    return pmin * r ** np.arange(n + 1)


def bls_run(t, f, e, periods, durations_d):
    from astropy.timeseries import BoxLeastSquares

    bls = BoxLeastSquares(t, f, dy=e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = bls.power(periods, durations_d, objective="snr")
    return bls, res


def sde(power):
    p = np.asarray(power)
    return (p - np.mean(p)) / np.std(p)


# ---------------------------------------------------------------- periodic search
def periodic_search(t, f, e, sec, pmin, pmax, n_signals=3, min_transits=2, oversample=3):
    durs = np.array([d for d in DURATIONS_H if d / 24 < 0.5 * pmin]) / 24
    found, mask = [], np.ones(len(t), bool)
    # rejected peaks (single events, artifacts) are masked and do not use up one of the n_signals slots
    for _ in range(n_signals + 5):
        if sum(not s["rejected"] for s in found) >= n_signals:
            break
        peaks = []
        for a, b in seasons(t[mask], sec[mask]):
            m = mask & (t >= a) & (t <= b)
            base = b - a
            pmax_s = min(pmax, base / min_transits)
            if pmax_s <= pmin or m.sum() < 100:
                continue
            P = period_grid(pmin, pmax_s, base, durs.min(), oversample)
            _, res = bls_run(t[m], f[m], e[m], P, durs)
            z = sde(res.power)
            i = int(np.argmax(res.power))
            peaks.append({"period": float(res.period[i]), "sde": float(z[i]), "season": (float(a), float(b))})
        if not peaks:
            break
        # cluster season peaks by period (0.5%) and pick the cluster seen in most seasons, then highest SDE
        best = max(peaks, key=lambda p: (sum(abs(q["period"] / p["period"] - 1) < 0.005 for q in peaks), p["sde"]))
        members = [q for q in peaks if abs(q["period"] / best["period"] - 1) < 0.005]
        sig = refine(t[mask], f[mask], e[mask], best["period"], durs, oversample)
        sig.update(n_seasons=len(members), sde=round(max(q["sde"] for q in members), 2),
                   season_sde=[round(q["sde"], 2) for q in members])
        # a "periodic" peak built from fewer than min_transits events is a single dip or an artifact
        sig["rejected"] = (f"only {sig['n_transits']} transit(s) with data" if sig["n_transits"] < min_transits else
                           f"one event carries {sig['top_event_power_frac']:.0%} of the signal" if sig["top_event_power_frac"] > 0.6
                           else None)
        found.append(sig)
        mask &= np.abs(((t - sig["epoch_btjd"] + 0.5 * sig["period_d"]) % sig["period_d"]) - 0.5 * sig["period_d"]) \
            > 1.0 * sig["duration_h"] / 24
    return found


def refine(t, f, e, p0, durs, oversample):
    """Phase-coherent fine grid around p0 on the full baseline, then BLS statistics."""
    base = t.max() - t.min()
    P = period_grid(p0 * 0.995, p0 * 1.005, base, durs.min(), oversample)
    bls, res = bls_run(t, f, e, P, durs)
    i = int(np.argmax(res.power))
    p, t0, d = float(res.period[i]), float(res.transit_time[i]), float(res.duration[i])
    st = bls.compute_stats(p, d, t0)
    t0 = t0 + p * np.round((np.median(t) - t0) / p)  # epoch near the middle of the data
    dep, dep_e = st["depth"]
    (do, doe), (de, dee) = st["depth_odd"], st["depth_even"]
    n_tr = int(np.sum(st["per_transit_count"] > 0))
    ll = np.asarray(st["per_transit_log_likelihood"], float)
    pos = ll[ll > 0]
    top_frac = float(pos.max() / pos.sum()) if len(pos) else 1.0
    return {"kind": "periodic", "period_d": round(p, 6), "epoch_btjd": round(float(t0), 5),
            "duration_h": round(d * 24, 3), "depth_ppm": round(dep * 1e6, 1), "depth_err_ppm": round(dep_e * 1e6, 1),
            "snr": round(dep / dep_e, 2) if dep_e > 0 else None, "n_transits": n_tr,
            "odd_depth_ppm": round(do * 1e6, 1), "even_depth_ppm": round(de * 1e6, 1),
            "odd_even_sigma": round(abs(do - de) / np.hypot(doe, dee), 2) if np.hypot(doe, dee) > 0 else None,
            "sinusoid_like": bool(st["harmonic_delta_log_likelihood"] < 0),
            "top_event_power_frac": round(top_frac, 3)}


# ---------------------------------------------------------------- single transits
def _segments(t, gap_d=0.5):
    cut = np.where(np.diff(t) > gap_d)[0]
    return list(zip(np.r_[t[0], t[cut + 1]], np.r_[t[cut], t[-1]]))


def single_transits(t, f, sec, snr_min=7.0, durations_h=SINGLE_DURATIONS_H):
    cands = []
    for dh in durations_h:
        dur = dh / 24
        rows = []
        for s in np.unique(sec):
            m = sec == s
            ts, fs = t[m], f[m]
            sig = robust_std(fs)
            for tc in np.arange(ts.min() + dur, ts.max() - dur, dur / 4):
                dt = ts - tc
                inn = np.abs(dt) <= dur / 2
                pre = (dt < -dur / 2) & (dt > -2 * dur)
                post = (dt > dur / 2) & (dt < 2 * dur)
                if inn.sum() < 4 or pre.sum() < 3 or post.sum() < 3:
                    continue
                base = np.median(fs[pre | post])
                depth = base - np.mean(fs[inn])
                edge = min(abs(tc - a) for seg in _segments(ts) for a in seg) * 24
                rows.append((tc, depth, depth / (sig / np.sqrt(inn.sum())), s, int(inn.sum()), edge))
        if not rows:
            continue
        raw = np.array([r[2] for r in rows])
        norm = robust_std(raw) or 1.0      # red-noise normalization
        for (tc, depth, snr_raw, s, n, edge) in rows:
            snr = snr_raw / max(norm, 1.0)
            if snr >= snr_min:
                cands.append({"kind": "single", "epoch_btjd": round(float(tc), 4), "duration_h": dh,
                              "depth_ppm": round(depth * 1e6, 1), "snr": round(float(snr), 2), "sector": int(s),
                              "n_in": n, "noise_scale": round(float(norm), 2), "hours_to_gap_or_edge": round(edge, 1),
                              "near_gap_or_edge": bool(edge < 2 * dh)})
    # keep the best box per dip (non-overlapping)
    cands.sort(key=lambda c: -c["snr"])
    kept = []
    for c in cands:
        if all(abs(c["epoch_btjd"] - k["epoch_btjd"]) > 0.5 * (c["duration_h"] + k["duration_h"]) / 24 for k in kept):
            kept.append(c)
    return kept


# ---------------------------------------------------------------- main
def search(tic, sectors=None, lc_files=None, cache_dir=None, pmin=1.0, pmax=None, n_signals=3, single_snr=7.0,
           window_d=1.0, bin_min=10.0, out_dir=None) -> dict:
    tic = parse_tic(tic)
    secs, load = load_sectors(tic, cache_dir, lc_files)
    if sectors:
        secs = [s for s in secs if s["sector"] in set(sectors)]
    if not secs:
        raise RuntimeError("no light curves to search")
    prepare_sectors(secs, None, 0.0, 0.0, window_d=window_d)
    t, f, e, sec = binned_flat(secs, bin_min)
    pmax = pmax or max(pmin * 2, min(100.0, max(b - a for a, b in seasons(t, sec)) / 2))
    periodic = periodic_search(t, f, e, sec, pmin, pmax, n_signals)
    singles = single_transits(t, f, sec, single_snr)
    for s in singles:
        s["matches_periodic"] = None
        for p in periodic:
            ph = ((s["epoch_btjd"] - p["epoch_btjd"] + 0.5 * p["period_d"]) % p["period_d"]) - 0.5 * p["period_d"]
            if abs(ph) < 0.5 * (s["duration_h"] + p["duration_h"]) / 24:
                s["matches_periodic"] = p["period_d"]
    for i, p in enumerate(periodic):
        h = period_harmonic_flag(p["period_d"])
        p["harmonic_flag"] = h["flagged"]
        p["tess_orbit_flag"] = {r["ref_d"]: r["offset_pct"] for r in h["refs"]}
        p["harmonic_of"] = [q["period_d"] for j, q in enumerate(periodic) if j != i and
                            any(abs(p["period_d"] / q["period_d"] - r) < 0.01 for r in (2, 3, 0.5, 1 / 3))]
    rejected = [p for p in periodic if p["rejected"]]
    periodic = [p for p in periodic if not p["rejected"]]
    for s in singles:  # singles that only matched a rejected "periodic" artifact stay singles
        if s["matches_periodic"] and s["matches_periodic"] not in [p["period_d"] for p in periodic]:
            s["matches_periodic"] = None
    # singles are always listed; matches_periodic says when a dip is just one transit of a periodic signal
    signals = sorted(periodic, key=lambda p: -(p["sde"] or 0)) + [s for s in singles if not s["matches_periodic"]] \
        + [s for s in singles if s["matches_periodic"]][:20]
    for rank, s in enumerate(signals, 1):
        s["rank"] = rank
        s["status"] = "unpromoted"
        s["next_step"] = "run allsector_gate + catalog_veto before calling this a candidate"
    result = {"tool": "transit_search", "tic": tic, "generated_at": now_et(),
              "inputs": {"sectors": [s["sector"] for s in secs], "pmin": pmin, "pmax": round(pmax, 3),
                         "n_signals": n_signals, "single_snr": single_snr, "window_d": window_d, "bin_min": bin_min},
              "n_points_binned": int(len(t)), "signals": signals,
              "singles_matching_periodic": [s for s in singles if s["matches_periodic"]],
              "rejected_periodic": rejected, "load": load,
              "note": "Search output only. Every signal is unpromoted until allsector_gate has run; no detection is claimed."}
    out_dir = Path(out_dir or Path(cache_dir or default_cache_dir()) / "transit_search" / f"TIC{tic}")
    write_json(out_dir / "signals.json", result)
    with (out_dir / "signals.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["tic"] + CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for s in signals:
            w.writerow(jsonable({"tic": tic, **s, "tess_orbit_flag": s.get("tess_orbit_flag") and json.dumps(s["tess_orbit_flag"])}))
    result["json"], result["csv"] = str(out_dir / "signals.json"), str(out_dir / "signals.csv")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tic")
    ap.add_argument("--sectors", type=lambda s: [int(x) for x in s.split(",") if x])
    ap.add_argument("--lc-file", action="append")
    ap.add_argument("--pmin", type=float, default=1.0)
    ap.add_argument("--pmax", type=float)
    ap.add_argument("--n-signals", type=int, default=3)
    ap.add_argument("--single-snr", type=float, default=7.0)
    ap.add_argument("--window", type=float, default=1.0, help="biweight window, days")
    ap.add_argument("--bin-min", type=float, default=10.0)
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args(argv)
    r = search(a.tic, a.sectors, a.lc_file, a.cache_dir, a.pmin, a.pmax, a.n_signals, a.single_snr, a.window, a.bin_min,
               a.out_dir)
    print(json.dumps(jsonable({"tic": r["tic"], "inputs": r["inputs"], "json": r["json"], "csv": r["csv"],
                               "signals": [{k: s.get(k) for k in ["rank"] + CSV_COLS[1:]} for s in r["signals"][:10]]}),
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
