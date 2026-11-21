"""cross_star_veto: flag dips that happen at the same time on many unrelated stars.

Stars on one sector/camera/CCD share the spacecraft: momentum dumps, scattered light, pointing
jitter and heater cycles. A dip that recurs on more than N stars (or more than a fraction of the
stars searched) within tol_d is common-mode, not astrophysical.

Input is a dip table (tic, t0_btjd, plus anything else: sector, snr, depth_ppm, duration_h).
It can come from your own search, or from light curves via dips_from_lightcurves(), which applies
masks.cadence_mask, detrends, and runs transit_search.single_transits on each star.

Output:
  per-dip   n_stars_at_epoch (this star + distinct other stars with a dip within tol_d),
            other_tics, common_mode (n_stars_at_epoch > max_stars or > max_frac * n_stars_total)
  epochs    CCD-level table of common-mode epochs: clusters of dip times (single linkage at tol_d)
            with t_mid, t_min, t_max, n_stars, frac, tics

    cross-star-veto --dips dips.csv --tol 1.0 --max-stars 1
    cross-star-veto --tics 1,2,3 --sector 74 --out-dir veto_s74     # fetches each star's light curve
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import default_cache_dir, jsonable, now_et, parse_tic, write_json
from .masks import cadence_mask, summarize

TOL_D = 0.5
MAX_STARS = 2


def cross_star_veto(dips: pd.DataFrame, tol_d=TOL_D, max_stars=MAX_STARS, max_frac=None, n_stars_total=None):
    """Return (dips with veto columns, common-mode epoch table). Pure function, no I/O."""
    d = dips.copy().reset_index(drop=True)
    d["tic"] = d["tic"].astype(int)
    t, tic = d["t0_btjd"].to_numpy(float), d["tic"].to_numpy()
    n_total = n_stars_total or d["tic"].nunique()
    limit = max_stars if max_frac is None else min(max_stars, max_frac * n_total)
    order = np.argsort(t)
    ts, tics_sorted = t[order], tic[order]
    others, counts = [], []
    for ti, ci in zip(t, tic):
        lo, hi = np.searchsorted(ts, ti - tol_d, "left"), np.searchsorted(ts, ti + tol_d, "right")
        o = sorted({int(x) for x in tics_sorted[lo:hi] if x != ci})
        others.append(o)
        counts.append(1 + len(o))
    d["n_stars_at_epoch"] = counts
    d["other_tics"] = [";".join(map(str, o)) for o in others]
    d["common_mode"] = d["n_stars_at_epoch"] > limit
    # CCD-level table: single-linkage clusters of all dip times
    cid = np.zeros(len(t), int)
    if len(t):
        cid_sorted = np.r_[0, np.cumsum(np.diff(ts) > tol_d)]
        cid[order] = cid_sorted
    d["epoch_cluster"] = cid
    rows = []
    for c, g in d.groupby("epoch_cluster"):
        n = g["tic"].nunique()
        if n > limit:
            rows.append({"epoch_cluster": int(c), "t_mid_btjd": round(float(g["t0_btjd"].median()), 4),
                         "t_min_btjd": round(float(g["t0_btjd"].min()), 4), "t_max_btjd": round(float(g["t0_btjd"].max()), 4),
                         "n_stars": int(n), "frac_stars": round(n / n_total, 4), "n_dips": int(len(g)),
                         "sectors": ";".join(map(str, sorted(g["sector"].unique()))) if "sector" in g else "",
                         "tics": ";".join(map(str, sorted(g["tic"].unique())))})
    epochs = pd.DataFrame(rows, columns=["epoch_cluster", "t_mid_btjd", "t_min_btjd", "t_max_btjd", "n_stars", "frac_stars",
                                         "n_dips", "sectors", "tics"])
    return d, epochs.sort_values("n_stars", ascending=False).reset_index(drop=True)


def flag_events(events: pd.DataFrame, dips: pd.DataFrame, tol_d=TOL_D, max_stars=MAX_STARS, max_frac=None,
                n_stars_total=None):
    """Veto arbitrary events (e.g. candidate epochs) against a dip table: same rule, events need not be in dips."""
    n_total = n_stars_total or dips["tic"].nunique()
    limit = max_stars if max_frac is None else min(max_stars, max_frac * n_total)
    t, tic = dips["t0_btjd"].to_numpy(float), dips["tic"].to_numpy(int)
    e = events.copy()
    o = [sorted({int(x) for x in tic[(np.abs(t - te) <= tol_d) & (tic != int(ce))]})
         for te, ce in zip(e["t0_btjd"].astype(float), e["tic"])]
    e["n_stars_at_epoch"] = [1 + len(x) for x in o]
    e["other_tics"] = [";".join(map(str, x)) for x in o]
    e["common_mode"] = e["n_stars_at_epoch"] > limit
    return e


def dips_from_lightcurves(lcs, snr_min=7.0, window_d=1.0, bin_min=30.0, mask_kw=None, edge_policy="flag",
                          edge_snr_min=10.0):
    """lcs: list of sector dicts (allsector_gate.read_lc_file format) with a 'tic' key.

    Each light curve is masked (masks.cadence_mask), biweight-detrended and searched for single dips.
    edge_policy: "drop" removes the first/last edge_d of every orbit before searching. "flag" (default)
    searches them too, but a dip centred there is kept only at SNR >= edge_snr_min and carries
    in_orbit_edge=True. Real transits do land near orbit ends (TOI-4600 c in S17 ends 0.4 d before
    the orbit does), so dropping edges outright loses them.
    Returns (dip DataFrame, per-star mask summary).
    """
    from .allsector_gate import detrend
    from .masks import EDGE_D, stretches
    from .transit_search import binned_flat, single_transits

    kw = dict(mask_kw or {})
    edge_d = kw.pop("edge_d", EDGE_D)
    rows, info = [], []
    for lc in lcs:
        args = (lc["time"], lc["quality"], lc["flux"], lc.get("author", "SPOC"))
        good_edge, why = cadence_mask(*args, edge_d=edge_d, **kw)
        good = good_edge if edge_policy == "drop" else cadence_mask(*args, edge_d=0, **kw)[0]
        s = dict(lc, good=good)
        s["flat"] = detrend(s["time"], s["flux"], good, np.zeros(len(good), bool), window_d)
        info.append({"tic": lc["tic"], "sector": lc["sector"], "n_cadences": len(good), "n_good": int(good_edge.sum()),
                     "edge_policy": edge_policy, **{f"masked_{k}": v for k, v in summarize(why).items()}})
        if good.sum() < 50:
            continue
        fin = np.isfinite(lc["time"]) & np.isfinite(lc["flux"])
        bounds = stretches(lc["time"][fin])
        t, f, _, sec = binned_flat([s], bin_min)
        for dip in single_transits(t, f, sec, snr_min):
            tc = dip["epoch_btjd"]
            in_edge = any((a <= tc < a + edge_d) or (b - edge_d < tc <= b) for a, b in bounds)
            if in_edge and dip["snr"] < edge_snr_min:
                continue
            rows.append({"tic": int(lc["tic"]), "sector": dip["sector"], "t0_btjd": tc,
                         "duration_h": dip["duration_h"], "depth_ppm": dip["depth_ppm"], "snr": dip["snr"],
                         "near_gap_or_edge": dip["near_gap_or_edge"], "in_orbit_edge": bool(in_edge)})
    cols = ["tic", "sector", "t0_btjd", "duration_h", "depth_ppm", "snr", "near_gap_or_edge", "in_orbit_edge"]
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(info)


def load_sector_lcs(tics, sector, cache_dir=None):
    """One light curve per TIC for one sector via tess_lc_fetch (SPOC, then TESS-SPOC, then QLP)."""
    from .allsector_gate import read_lc_file
    from .tess_lc_fetch import fetch

    lcs, failures = [], []
    for tic in tics:
        try:
            _, m = fetch(tic, cache_dir=cache_dir, pick="best", sectors=[sector], load=False)
            for p in m["products"]:
                for s in read_lc_file(p["local_path"]):
                    lcs.append(dict(s, tic=int(tic)))
            if not m["products"]:
                failures.append({"tic": int(tic), "reason": "no product in sector"})
        except Exception as exc:  # noqa: BLE001 - one bad star must not stop the CCD
            failures.append({"tic": int(tic), "reason": f"{type(exc).__name__}: {exc}"[:200]})
    return lcs, failures


def run(dips=None, tics=None, sector=None, tol_d=TOL_D, max_stars=MAX_STARS, max_frac=None, snr_min=7.0, out_dir=None,
        cache_dir=None, lcs=None):
    failures, mask_info = [], None
    if dips is None:
        if lcs is None:
            lcs, failures = load_sector_lcs(tics, sector, cache_dir)
        dips, mask_info = dips_from_lightcurves(lcs, snr_min)
        n_total = len({lc["tic"] for lc in lcs})
    else:
        n_total = None
    flagged, epochs = cross_star_veto(dips, tol_d, max_stars, max_frac, n_total)
    out_dir = Path(out_dir or Path(cache_dir or default_cache_dir()) / "cross_star_veto" /
                   (f"s{sector}" if sector else "dips"))
    out_dir.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(out_dir / "dips_flagged.csv", index=False)
    epochs.to_csv(out_dir / "common_mode_epochs.csv", index=False)
    if mask_info is not None:
        mask_info.to_csv(out_dir / "mask_summary.csv", index=False)
    summary = {"tool": "cross_star_veto", "generated_at": now_et(),
               "params": {"tol_d": tol_d, "max_stars": max_stars, "max_frac": max_frac, "snr_min": snr_min, "sector": sector},
               "n_stars": int(n_total or flagged["tic"].nunique()), "n_dips": int(len(flagged)),
               "n_common_mode_dips": int(flagged["common_mode"].sum()), "n_common_mode_epochs": int(len(epochs)),
               "top_epochs": epochs.head(10).to_dict("records"), "fetch_failures": failures,
               "files": {"dips": str(out_dir / "dips_flagged.csv"), "epochs": str(out_dir / "common_mode_epochs.csv")}}
    write_json(out_dir / "summary.json", summary)
    return summary, flagged, epochs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dips", type=Path, help="CSV with tic,t0_btjd[,sector,snr,...]")
    ap.add_argument("--tics", type=lambda s: [parse_tic(x) for x in s.split(",") if x])
    ap.add_argument("--sector", type=int)
    ap.add_argument("--events", type=Path, help="optional CSV of events (tic,t0_btjd) to veto against the dips")
    ap.add_argument("--tol", type=float, default=TOL_D, help="days")
    ap.add_argument("--max-stars", type=int, default=MAX_STARS, help="flag when the dip is on more than this many stars")
    ap.add_argument("--max-frac", type=float)
    ap.add_argument("--snr-min", type=float, default=7.0)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--cache-dir", type=Path)
    a = ap.parse_args(argv)
    if a.dips is None and not (a.tics and a.sector):
        ap.error("give --dips, or --tics with --sector")
    dips = pd.read_csv(a.dips) if a.dips else None
    s, _, _ = run(dips, a.tics, a.sector, a.tol, a.max_stars, a.max_frac, a.snr_min, a.out_dir, a.cache_dir)
    if a.events is not None:
        ev = flag_events(pd.read_csv(a.events), pd.read_csv(s["files"]["dips"]), a.tol, a.max_stars, a.max_frac)
        p = Path(s["files"]["dips"]).with_name("events_flagged.csv")
        ev.to_csv(p, index=False)
        s["events"] = {"file": str(p), "n": len(ev), "n_common_mode": int(ev["common_mode"].sum())}
    print(json.dumps(jsonable(s), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
