"""bls_triage: the CCD-wide cuts that turn thousands of BLS outputs into a short gate list.

These are the cuts from the S74/1/2 hunt (21 Nov 2026), plus the two checks that hunt had to do by hand:

  snr            BLS SNR >= 9
  n_transits     >= 3 transits with data
  odd_even       odd/even depth difference < 3 sigma
  dominance      top signal SNR >= 1.5x the star's next-best periodic signal
  harmonic       not within 5% of the 6.85 / 13.7 / 27.4 d TESS orbit harmonics
  integer_day    |P - round(P)| >= 0.03 d (for P >= 0.9 d)
  radius         Rp = sqrt(depth) * R* < 20 Re
  grid_edge      not (duration pinned at the top of the duration grid and P < 3 d): variability
  duration       duration_check: T14 possible for the TIC density (catches grazing EBs, e.g. TIC 141618383)
  systematic     fewer than 1/3 of the transits land in CCD-wide "bad windows": 0.5-d bins where many
                 stars on the CCD have a single dip at once (the BTJD 2744-2751 cluster in S53)

Only each star's top periodic signal is triaged (`all_signals=True` for every one). A signal that fails
exactly one soft cut (snr >= 7, dominance, n_transits, systematic) is a near miss; survivors and near
misses go to allsector_gate. Nothing here is a detection.

    bls-triage --run-dir runs/s74_c1_d2            # reads <run>/stages/bls/*.json written by `hunt`
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .common import jsonable, write_json
from .duration_check import check as duration_check

CUTS = {"snr_min": 9.0, "near_snr_min": 7.0, "min_transits": 3, "odd_even_max": 3.0, "dominance_min": 1.5,
        "integer_day_tol": 0.03, "rp_max_re": 20.0, "edge_pmax": 3.0, "systematic_frac": 1 / 3,
        "bad_bin_d": 0.5, "bad_min_stars": 5, "bad_factor": 5.0}
SOFT = {"snr", "dominance", "n_transits", "systematic"}
RSUN_RE = 109.08
COLS = ["tic", "status", "fails", "period_d", "epoch_btjd", "duration_h", "depth_ppm", "snr", "n_transits",
        "odd_even_sigma", "dominance", "rp_re", "t14_ratio", "implied_b", "frac_bad", "rank"]


def bad_windows(stars, bin_d=0.5, min_stars=5, factor=5.0):
    """Times where many stars dip at once. stars: iterable of dicts with 'tic' and 'singles' [{epoch_btjd}]."""
    per_bin = {}
    for s in stars:
        for b in {math.floor(d["epoch_btjd"] / bin_d) for d in s.get("singles", [])}:
            per_bin.setdefault(b, set()).add(s["tic"])
    if not per_bin:
        return [], {}
    counts = {b: len(v) for b, v in per_bin.items()}
    thr = max(min_stars, factor * float(np.median(list(counts.values()))))
    bad = sorted(b for b, c in counts.items() if c >= thr)
    wins = []
    for b in bad:
        if wins and b * bin_d <= wins[-1][1] + 1e-9:
            wins[-1][1] = (b + 1) * bin_d
            wins[-1][2] = max(wins[-1][2], counts[b])
        else:
            wins.append([b * bin_d, (b + 1) * bin_d, counts[b]])
    return [{"start_btjd": round(a, 3), "end_btjd": round(e, 3), "max_stars": c} for a, e, c in wins], {"threshold": thr}


def transit_times(sig, cover):
    """Predicted mid-times that fall inside the data coverage."""
    P, T0 = sig["period_d"], sig["epoch_btjd"]
    out = []
    for a, b in cover:
        n0, n1 = math.ceil((a - T0) / P), math.floor((b - T0) / P)
        out += [T0 + n * P for n in range(n0, n1 + 1)]
    return out


def frac_in_bad(times, wins, pad_d=0.0):
    if not times or not wins:
        return 0.0
    hit = sum(any(w["start_btjd"] - pad_d <= t <= w["end_btjd"] + pad_d for w in wins) for t in times)
    return hit / len(times)


def triage_star(star, wins, cuts=None, all_signals=False):
    """Rows for one star's periodic signals. star: {tic, rstar, rho_solar, signals, coverage}."""
    c = {**CUTS, **(cuts or {})}
    per = sorted([s for s in star.get("signals", []) if s.get("kind", "periodic") == "periodic"],
                 key=lambda s: -(s.get("snr") or 0))
    rows = []
    for i, s in enumerate(per if all_signals else per[:1]):
        fails = []
        snr = s.get("snr") or 0.0
        nxt = max([(o.get("snr") or 0.0) for j, o in enumerate(per) if j != i] or [0.0])
        dom = snr / nxt if nxt > 0 else float("inf")
        P, dep = s["period_d"], max(s.get("depth_ppm") or 0.0, 0.0)
        rstar, rho = star.get("rstar"), star.get("rho_solar")
        rp = math.sqrt(dep * 1e-6) * rstar * RSUN_RE if rstar and np.isfinite(rstar) else None
        if snr < c["snr_min"]:
            fails.append("snr")
        if (s.get("n_transits") or 0) < c["min_transits"]:
            fails.append("n_transits")
        oe = s.get("odd_even_sigma")
        if oe is not None and oe >= c["odd_even_max"]:
            fails.append("odd_even")
        if i == 0 and dom < c["dominance_min"]:
            fails.append("dominance")
        if s.get("harmonic_flag"):
            fails.append("harmonic")
        if P >= 0.9 and abs(P - round(P)) < c["integer_day_tol"]:
            fails.append("integer_day")
        if rp is not None and rp >= c["rp_max_re"]:
            fails.append("radius")
        if s.get("duration_at_grid_edge") and P < c["edge_pmax"]:
            fails.append("grid_edge")
        dc = duration_check(P, s["duration_h"], dep, rho if rho and np.isfinite(rho) else None)
        if dc["result"] == "fail":
            fails.append("duration")
        fb = frac_in_bad(transit_times(s, star.get("coverage") or []), wins)
        if fb >= c["systematic_frac"]:
            fails.append("systematic")
        near = len(fails) == 1 and fails[0] in SOFT and (fails[0] != "snr" or snr >= c["near_snr_min"])
        rows.append({"tic": int(star["tic"]), "status": "survivor" if not fails else ("near_miss" if near else "cut"),
                     "fails": ";".join(fails), "period_d": P, "epoch_btjd": s["epoch_btjd"], "duration_h": s["duration_h"],
                     "depth_ppm": s.get("depth_ppm"), "snr": snr, "n_transits": s.get("n_transits"), "odd_even_sigma": oe,
                     "dominance": None if not math.isfinite(dom) else round(dom, 2), "rp_re": None if rp is None else round(rp, 2),
                     "t14_ratio": dc.get("ratio_obs_over_b0"), "implied_b": dc.get("implied_b"), "frac_bad": round(fb, 2),
                     "rank": i + 1, "duration_flag": dc.get("flag")})
    return rows


def triage(stars, cuts=None, all_signals=False, out_dir=None) -> dict:
    """stars: list of {tic, rstar, rho_solar, signals, singles, coverage}. Returns counts, survivors, near misses."""
    c = {**CUTS, **(cuts or {})}
    stars = [s for s in stars if s.get("ok", True)]
    wins, info = bad_windows(stars, c["bad_bin_d"], c["bad_min_stars"], c["bad_factor"])
    rows = [r for s in stars for r in triage_star(s, wins, c, all_signals)]
    fail_counts = {}
    for r in rows:
        for f in filter(None, r["fails"].split(";")):
            fail_counts[f] = fail_counts.get(f, 0) + 1
    key = lambda r: -(r["snr"] or 0)
    res = {"tool": "bls_triage", "cuts": c, "n_stars": len(stars), "n_signals": len(rows),
           "n_survivors": sum(r["status"] == "survivor" for r in rows),
           "n_near_miss": sum(r["status"] == "near_miss" for r in rows), "fail_counts": fail_counts,
           "bad_windows": wins, "bad_window_threshold": info.get("threshold"),
           "survivors": sorted([r for r in rows if r["status"] == "survivor"], key=key),
           "near_misses": sorted([r for r in rows if r["status"] == "near_miss"], key=key)}
    if out_dir:
        out_dir = Path(out_dir)
        write_json(out_dir / "triage.json", res)
        with (out_dir / "triage.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(jsonable(sorted(rows, key=key)))
        res["csv"] = str(out_dir / "triage.csv")
    return res


def load_run(run_dir) -> list[dict]:
    """BLS stage files written by `hunt`, joined with the TIC parameters from its select stage."""
    run_dir = Path(run_dir)
    params = {}
    for p in (run_dir / "stages" / "select").glob("*.json"):
        for r in json.loads(p.read_text()).get("selected", []):
            params[int(r["tic"])] = r
    stars = []
    for p in sorted((run_dir / "stages" / "bls").glob("TIC*.json")):
        d = json.loads(p.read_text())
        if not d.get("ok"):
            continue
        tp = params.get(int(d["tic"]), {})
        stars.append({**d, "rstar": d.get("rstar", tp.get("rstar")), "rho_solar": d.get("rho_solar", tp.get("rho_solar"))})
    return stars


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--all-signals", action="store_true")
    for k, v in CUTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args(argv)
    r = triage(load_run(a.run_dir), {k: getattr(a, k) for k in CUTS}, a.all_signals, a.run_dir)
    print(f"stars={r['n_stars']} survivors={r['n_survivors']} near_miss={r['n_near_miss']} "
          f"bad_windows={len(r['bad_windows'])} fails={json.dumps(r['fail_counts'])} -> {r.get('csv')}")
    for x in (r["survivors"] + r["near_misses"])[:25]:
        print(f"  {x['status']:9s} TIC{x['tic']} P={x['period_d']:.5f} snr={x['snr']:.1f} depth={x['depth_ppm']} "
              f"{x['fails'] or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
