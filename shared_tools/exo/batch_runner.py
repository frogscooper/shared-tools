"""batch_runner: resumable duo-transit funnel over a list of TICs or one sector/camera/CCD.

Stages (each writes <run>/stages/<stage>/TIC<id>.json; existing files are reused, so a crashed or
interrupted run picks up where it stopped; --redo STAGE recomputes one stage):

  select    star list (--tics, or the TESS-SPOC target list for --sector mapped to --camera/--ccd with
            tess-point), TIC cuts (defaults: Tmag 8-13, Teff 3500-6000 K, logg >= 4.0), and the
            known-object prefilter (TIC in today's ExoFOP TOI or CTOI table)
  coverage  SPOC / TESS-SPOC light curves exist in an early sector (1-26) and a late one (>= 40)
  lightcurves  one product per sector (tess_lc_fetch); min CROWDSAP >= 0.9
  dips      masks.cadence_mask, biweight detrend, transit_search.single_transits per sector
  veto      cross_star_veto per sector over every star in the run (common-mode dips removed);
            needs >= 10 stars to mean anything, so control runs skip it
  pairs     dips matched across years: |dT| >= 200 d, depth ratio 0.5-2, duration ratio 0.5-2
            (with --sector, one of the two dips must be in that sector)
  duo       duo_solver on each pair (aliases checked against all coverage + the duo gate)
  packet    candidate_packet duo packet for every pair that passes the duo gate: per-event centroid,
            dilution, stellar density per alias, catalog_veto

funnel.csv counts what survives each stage. Nothing is submitted anywhere; a "duo candidate" is a
hypothesis for a human to review.

    batch-runner --control toi4600c --run-dir runs/control_toi4600c
    batch-runner --sector 74 --camera 1 --ccd 2 --run-dir runs/s74_c1_d2 --max-stars 50
    batch-runner --tics 232608943,209464063 --run-dir runs/mine --no-known-filter
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .common import default_cache_dir, jsonable, now_et, parse_tic, write_json

STAGES = ["select", "coverage", "lightcurves", "dips", "veto", "pairs", "duo", "packet"]
CUTS = {"tmag_min": 8.0, "tmag_max": 13.0, "teff_min": 3500.0, "teff_max": 6000.0, "logg_min": 4.0,
        "crowdsap_min": 0.9, "early_max_sector": 26, "late_min_sector": 40}
CONTROLS = {
    # Mireles+2023: TOI-4600 c, P = 482.8191 d; TESS transits in S17 and S53
    "toi4600c": {"tic": 232608943, "period_d": 482.8191, "t1": 1785.96, "t2": 2751.5986, "name": "TOI-4600 c"},
}
MIN_STARS_FOR_VETO = 10


# ---------------------------------------------------------------- helpers
class Deferred(Exception):
    """The run's time budget ran out before this stage item started. Re-run to continue (all work is cached)."""


class Run:
    def __init__(self, run_dir: Path, redo=(), cache_dir=None, deadline=None):
        self.dir = Path(run_dir)
        self.redo = set(redo or ())
        self.cache_dir = Path(cache_dir or default_cache_dir())
        self.deadline = deadline          # time.time() after which no new (uncached) work starts
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, stage, key):
        return self.dir / "stages" / stage / f"{key}.json"

    def cached(self, stage, key, fn):
        """Return the stage result for key, computing (and saving) it only if missing or redone."""
        p = self.path(stage, key)
        if p.exists() and stage not in self.redo:
            return json.loads(p.read_text())
        if self.deadline is not None and time.time() > self.deadline:
            raise Deferred(f"{stage}/{key}")
        try:
            out = fn()
        except Exception as exc:  # noqa: BLE001 - one star must not kill the run; the error is the result
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500], "trace": traceback.format_exc()[-2000:]}
        write_json(p, out)
        return out


def tic_params(tics):
    """Tmag, Teff, logg, radius, mass, rho for TIC ids from the MAST TIC (batched)."""
    from astroquery.mast import Catalogs

    rows = []
    tics = [int(t) for t in tics]
    for i in range(0, len(tics), 500):
        tab = Catalogs.query_criteria(catalog="TIC", ID=tics[i:i + 500]).to_pandas()
        for _, r in tab.iterrows():
            rows.append({"tic": int(r["ID"]), "tmag": r.get("Tmag"), "teff": r.get("Teff"), "logg": r.get("logg"),
                         "rstar": r.get("rad"), "mstar": r.get("mass"), "rho_solar": r.get("rho"),
                         "ra": r.get("ra"), "dec": r.get("dec"), "lumclass": r.get("lumclass")})
    return pd.DataFrame(rows)


def apply_cuts(df, cuts):
    c = {**CUTS, **(cuts or {})}
    why = []
    for _, r in df.iterrows():
        w = []
        if not (c["tmag_min"] <= (r["tmag"] if pd.notna(r["tmag"]) else -99) <= c["tmag_max"]):
            w.append("tmag")
        if not (c["teff_min"] <= (r["teff"] if pd.notna(r["teff"]) else -99) <= c["teff_max"]):
            w.append("teff")
        if not ((r["logg"] if pd.notna(r["logg"]) else -99) >= c["logg_min"]):
            w.append("logg")
        why.append(";".join(w))
    out = df.copy()
    out["cut_fail"] = why
    return out


def known_tics(cache_dir):
    """TICs in today's ExoFOP TOI and CTOI tables (date-stamped cache from catalog_veto)."""
    from .catalog_veto import CTOI_URL, TOI_URL, _digits, exofop_table

    known = {}
    for name, url in (("toi", TOI_URL), ("ctoi", CTOI_URL)):
        df, _ = exofop_table(name, url, cache_dir)
        for v in df["TIC ID"].map(_digits):
            if v:
                known.setdefault(int(v), name)
    return known


def ccd_targets(sector, camera, ccd, cache_dir):
    """TESS-SPOC FFI target list for a sector, mapped to camera/CCD with tess-point."""
    from tess_stars2px import tess_stars2px_function_entry

    from .catalog_veto import http

    p = Path(cache_dir) / "target_lists" / f"tess-spoc_s{sector:04d}.csv"
    if not p.exists():
        r = http("GET", f"https://archive.stsci.edu/hlsps/tess-spoc/target_lists/s{sector:04d}.csv", timeout=(20, 300))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r.content)
    df = pd.read_csv(p)
    df.columns = ["tic", "ra", "dec"]
    out = tess_stars2px_function_entry(df["tic"].to_numpy(), df["ra"].to_numpy(), df["dec"].to_numpy(), trySector=sector)
    ids, cams, ccds = out[0], out[4], out[5]
    hit = set(int(i) for i, c1, c2 in zip(ids, cams, ccds) if c1 == camera and c2 == ccd)
    return sorted(hit)


# ---------------------------------------------------------------- stages
def stage_select(run, tics, sector, camera, ccd, cuts, known_filter, max_stars, controls):
    def work():
        ids = list(tics or [])
        source = "tics"
        if not ids:
            ids = ccd_targets(sector, camera, ccd, run.cache_dir)
            source = f"tess-spoc target list s{sector} cam{camera} ccd{ccd}"
        params = tic_params(ids)
        sel = apply_cuts(params, cuts)
        known = known_tics(run.cache_dir) if known_filter else {}
        sel["known"] = sel["tic"].map(lambda t: known.get(int(t), ""))
        sel["control"] = sel["tic"].isin([c["tic"] for c in controls])
        keep = sel[((sel["cut_fail"] == "") & (sel["known"] == "")) | sel["control"]].sort_values("tmag")
        if max_stars:
            keep = keep.head(max_stars)
        return {"ok": True, "source": source, "n_input": len(ids), "n_tic_rows": len(params),
                "n_pass_cuts": int((sel["cut_fail"] == "").sum()), "n_known_removed": int(((sel["cut_fail"] == "") &
                                                                                         (sel["known"] != "")).sum()),
                "cuts": {**CUTS, **(cuts or {})}, "known_filter": known_filter,
                "selected": jsonable(keep.to_dict("records")), "all": jsonable(sel.to_dict("records"))}
    key = "_".join(map(str, tics)) [:80] if tics else f"s{sector}_c{camera}_d{ccd}"
    return run.cached("select", f"select_{key}", work)


def stage_coverage(run, tic, cuts):
    c = {**CUTS, **(cuts or {})}

    def work():
        from .tess_lc_fetch import inventory

        rows, info = inventory(tic, cache_dir=run.cache_dir)
        secs = sorted({int(r["sector"]) for r in rows if r["author"].upper() in ("SPOC", "TESS-SPOC")})
        early = [s for s in secs if s <= c["early_max_sector"]]
        late = [s for s in secs if s >= c["late_min_sector"]]
        return {"ok": bool(early and late), "sectors": secs, "early": early, "late": late, "inventory": info}
    return run.cached("coverage", f"TIC{tic}", work)


def stage_lightcurves(run, tic, cuts):
    c = {**CUTS, **(cuts or {})}

    def work():
        from .allsector_gate import read_lc_file
        from .tess_lc_fetch import fetch

        _, m = fetch(tic, authors=("SPOC", "TESS-SPOC"), cache_dir=run.cache_dir, pick="best", load=False)
        files = [p["local_path"] for p in m["products"]]
        crowd = [s.get("crowdsap") for f in files for s in read_lc_file(f)]
        crowd = [x for x in crowd if x is not None]
        cmin = float(min(crowd)) if crowd else None
        return {"ok": bool(files) and (cmin is None or cmin >= c["crowdsap_min"]), "files": files,
                "crowdsap_min": cmin, "n_sectors": len(files), "failures": m["failures"]}
    return run.cached("lightcurves", f"TIC{tic}", work)


def stage_dips(run, tic, files, snr_min):
    def work():
        from .allsector_gate import read_lc_file
        from .cross_star_veto import dips_from_lightcurves

        lcs = [dict(s, tic=tic) for f in files for s in read_lc_file(f)]
        dips, info = dips_from_lightcurves(lcs, snr_min)
        return {"ok": True, "n_dips": len(dips), "dips": jsonable(dips.to_dict("records")),
                "mask_summary": jsonable(info.to_dict("records"))}
    return run.cached("dips", f"TIC{tic}", work)


FRAGMENT_TOL = 1.0   # dips closer than this many durations to a stronger dip are fragments of it
RP_MAX_RE = 25.0     # pairs implying a bigger body than this (sqrt(depth) * R*) are EB-sized, not paired
RSUN_RE = 109.08


def merge_fragments(dips, tol=FRAGMENT_TOL):
    """Drop dips that sit within tol x duration of a stronger dip on the same star.

    The box finder can report the two wings of one long eclipse as separate dips, and two
    wings from one event then pair up and pass the duo gate (TIC 99725941 in S74). The
    strongest dip keeps the event; the others are returned as fragments.
    """
    kept, frags = [], []
    for d in sorted(dips, key=lambda x: -x["snr"]):
        host = next((k for k in kept if int(k.get("tic", 0)) == int(d.get("tic", 0)) and
                     abs(d["t0_btjd"] - k["t0_btjd"]) <= tol * max(d["duration_h"], k["duration_h"]) / 24), None)
        if host is None:
            kept.append(d)
        else:
            frags.append({**d, "fragment_of": host["t0_btjd"]})
    return sorted(kept, key=lambda x: x["t0_btjd"]), frags


def match_pairs(dips, sector=None, min_dt=200.0, max_pairs=20, rstar_rsun=None, rp_max_re=RP_MAX_RE):
    d = sorted(dips, key=lambda x: x["t0_btjd"])
    pairs = []
    for i, a in enumerate(d):
        for b in d[i + 1:]:
            if b["t0_btjd"] - a["t0_btjd"] < min_dt:
                continue
            if sector is not None and sector not in (a["sector"], b["sector"]):
                continue
            rd = b["depth_ppm"] / a["depth_ppm"] if a["depth_ppm"] > 0 else np.inf
            rw = b["duration_h"] / a["duration_h"]
            if rstar_rsun and np.isfinite(rstar_rsun) and rp_max_re and \
                    np.sqrt(max(a["depth_ppm"], b["depth_ppm"]) * 1e-6) * rstar_rsun * RSUN_RE > rp_max_re:
                continue
            if 0.5 <= rd <= 2 and 0.5 <= rw <= 2:
                pairs.append({"t1": a["t0_btjd"], "t2": b["t0_btjd"], "s1": a["sector"], "s2": b["sector"],
                              "depth_ppm": round(float(np.sqrt(a["depth_ppm"] * b["depth_ppm"])), 1),
                              "duration_h": round(0.5 * (a["duration_h"] + b["duration_h"]), 2),
                              "snr": round(min(a["snr"], b["snr"]), 2)})
    return sorted(pairs, key=lambda p: -p["snr"])[:max_pairs]


def run_batch(run_dir, tics=None, sector=None, camera=None, ccd=None, cuts=None, known_filter=True, max_stars=None,
              control=None, snr_min=7.0, pmin=10.0, tol_d=0.5, max_veto_stars=2, redo=(), cache_dir=None, packets=True,
              deadline=None, pair_sector_rule=True):
    """pair_sector_rule=False pairs dips from any sectors (one pass instead of two) and labels each pair in_sector."""
    run = Run(run_dir, redo, cache_dir, deadline)
    controls = [CONTROLS[c] for c in (control or [])]
    tics = sorted(set(list(tics or []) + [c["tic"] for c in controls])) if (tics or controls) else None
    funnel = {"run_dir": str(run.dir), "started_at": now_et()}
    sel = stage_select(run, tics, sector, camera, ccd, cuts, known_filter and not controls, max_stars, controls)
    if not sel.get("ok"):
        raise RuntimeError(f"select failed: {sel.get('error')}")
    stars = sel["selected"]
    funnel.update(n_input=sel["n_input"], n_pass_tic_cuts=sel["n_pass_cuts"], n_known_removed=sel["n_known_removed"],
                  n_selected=len(stars))
    rho = {int(s["tic"]): s.get("rho_solar") for s in stars}
    cov = {s["tic"]: stage_coverage(run, int(s["tic"]), cuts) for s in stars}
    ok = [t for t, c in cov.items() if c.get("ok")]
    funnel["n_early_and_late_coverage"] = len(ok)
    lcs = {t: stage_lightcurves(run, int(t), cuts) for t in ok}
    ok = [t for t, c in lcs.items() if c.get("ok")]
    funnel["n_lightcurves_crowdsap_ok"] = len(ok)
    dips = {t: stage_dips(run, int(t), lcs[t]["files"], snr_min) for t in ok}
    all_dips = pd.DataFrame([d for t in ok for d in dips[t].get("dips", [])])
    funnel["n_stars_with_dips"] = sum(1 for t in ok if dips[t].get("n_dips"))
    funnel["n_dips"] = int(len(all_dips))

    # cross-star veto, per sector, over every star in the run
    def veto_work():
        from .cross_star_veto import cross_star_veto

        if len(all_dips) == 0 or len(ok) < MIN_STARS_FOR_VETO:
            return {"ok": True, "skipped": f"needs >= {MIN_STARS_FOR_VETO} stars, have {len(ok)}",
                    "dips": jsonable(all_dips.assign(common_mode=False).to_dict("records")) if len(all_dips) else []}
        parts, eps = [], []
        for s, g in all_dips.groupby("sector"):
            n_on = sum(1 for t in ok if any(d["sector"] == s for d in dips[t].get("dips", []))) or 1
            f, e = cross_star_veto(g, tol_d, max_veto_stars, None, max(n_on, len(ok)))
            parts.append(f)
            eps.append(e.assign(sector=s))
        f = pd.concat(parts)
        e = pd.concat(eps) if eps else pd.DataFrame()
        e.to_csv(run.dir / "common_mode_epochs.csv", index=False)
        return {"ok": True, "dips": jsonable(f.to_dict("records")), "n_common_mode": int(f["common_mode"].sum()),
                "n_epochs": int(len(e))}
    veto = run.cached("veto", "veto", veto_work)
    kept = [d for d in veto["dips"] if not d.get("common_mode")]
    funnel["n_dips_after_veto"] = len(kept)
    funnel["veto_note"] = veto.get("skipped", "")

    rstar = {int(s["tic"]): s.get("rstar") for s in stars}
    pairs = {}
    for t in ok:
        mine = [d for d in kept if int(d["tic"]) == int(t)]

        def pair_work(mine=mine, t=t):
            merged, frags = merge_fragments(mine)
            pp = match_pairs(merged, sector if pair_sector_rule else None, rstar_rsun=rstar.get(int(t)))
            for x in pp:
                x["in_sector"] = sector is None or sector in (x["s1"], x["s2"])
            return {"ok": True, "pairs": pp, "n_fragments_merged": len(frags), "fragments": frags}
        pairs[t] = run.cached("pairs", f"TIC{t}" if pair_sector_rule else f"TIC{t}_anysector", pair_work)
    funnel["n_dip_fragments_merged"] = sum(p.get("n_fragments_merged", 0) for p in pairs.values())
    funnel["n_pairs"] = sum(len(p["pairs"]) for p in pairs.values())
    funnel["n_stars_with_pairs"] = sum(1 for p in pairs.values() if p["pairs"])

    from .duo_solver import solve

    duo_rows = []
    for t in ok:
        for p in pairs[t]["pairs"]:
            key = f"TIC{t}_{p['t1']:.3f}_{p['t2']:.3f}"

            def duo_work(t=t, p=p):
                r = solve(int(t), p["t1"], p["t2"], p["duration_h"], p["depth_ppm"], pmin=pmin,
                          rho_solar=rho.get(int(t)) if rho.get(int(t)) and np.isfinite(rho.get(int(t))) else None,
                          cache_dir=run.cache_dir, lc_files=lcs[t]["files"], out_dir=run.dir / "duo" / key)
                return {"ok": True, "verdict": r["gate"]["verdict"], "label": r["label"],
                        "label_kind": r["gate"]["tests"]["label"]["result"],
                        "surviving_periods": r["surviving_periods"], "n_aliases": r["n_aliases"],
                        "n_excluded": r["n_excluded"], "events": r["events"], "json": r["json"],
                        "why": {k: v.get("why") for k, v in r["gate"]["tests"].items()}}
            r = run.cached("duo", key, duo_work)
            duo_rows.append({"tic": t, **p, **{k: r.get(k) for k in ("ok", "verdict", "label_kind", "surviving_periods",
                                                                     "n_excluded", "label", "error")}})
    funnel["n_duo_solved"] = sum(1 for r in duo_rows if r["ok"])
    # one pair shown to be periodic (a third dip seen) makes the star periodic: demote its other pairs
    periodic_stars = {int(r["tic"]) for r in duo_rows if r.get("label_kind") == "periodic"}
    for r in duo_rows:
        r["star_periodic"] = int(r["tic"]) in periodic_stars
    gate_pass = [r for r in duo_rows if r.get("verdict") == "pass" and r.get("label_kind") == "duo"]
    passed = [r for r in gate_pass if not r["star_periodic"]]
    funnel["n_duo_gate_pass"] = len(gate_pass)
    funnel["n_demoted_star_periodic"] = len(gate_pass) - len(passed)
    funnel["n_duo_gate_pass_after_demotion"] = len(passed)

    pk = []
    if packets:
        from .candidate_packet import build

        for r in passed:
            key = f"TIC{r['tic']}_{r['t1']:.3f}_{r['t2']:.3f}"

            def pk_work(r=r):
                sig = {"kind": "duo", "t1": r["t1"], "t2": r["t2"], "duration_h": r["duration_h"], "depth_ppm": r["depth_ppm"],
                       "pmin": pmin, "rho_solar": rho.get(int(r["tic"])), "source": "batch_runner"}
                s = build(int(r["tic"]), sig, run.cache_dir, lcs[r["tic"]]["files"], tpf_sectors=[r["s1"], r["s2"]],
                          out_root=run.dir / "packets")
                return {"ok": True, "status": s["status"], "packet_dir": s["packet_dir"],
                        "centroid": s["results"]["centroid_check"]["verdict"], "catalog": s["results"]["catalog_veto"]["verdict"]}
            pk.append(run.cached("packet", key, pk_work) | {"tic": r["tic"], "t1": r["t1"], "t2": r["t2"]})
    funnel["n_centroid_on_target"] = sum(1 for p in pk if p.get("centroid") == "on_target")
    funnel["n_catalog_clean"] = sum(1 for p in pk if p.get("catalog") == "clean")
    funnel["n_duo_candidates_for_review"] = sum(1 for p in pk if p.get("status") == "duo_candidate_needs_human_review")
    funnel["n_known_objects_recovered"] = sum(1 for p in pk if p.get("status") == "known_object")

    ctrl = []
    for c in controls:
        rows = [r for r in duo_rows if int(r["tic"]) == c["tic"]]
        hit = [r for r in rows if abs(r["t1"] - c["t1"]) < 0.5 and abs(r["t2"] - c["t2"]) < 0.5]
        rec = bool(hit) and any(abs(p - c["period_d"]) < 0.05 for p in (hit[0].get("surviving_periods") or []))
        ctrl.append({"control": c["name"], "tic": c["tic"], "pair_found": bool(hit), "true_period_survives": rec,
                     "surviving_periods": hit[0].get("surviving_periods") if hit else None,
                     "duo_gate": hit[0].get("verdict") if hit else None})
    funnel["finished_at"] = now_et()
    write_json(run.dir / "funnel.json", {"funnel": funnel, "controls": ctrl, "duo": duo_rows, "packets": pk})
    with (run.dir / "funnel.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "count"])
        for k, v in funnel.items():
            if k.startswith("n_"):
                w.writerow([k[2:], v])
    pd.DataFrame(duo_rows).to_csv(run.dir / "duo_pairs.csv", index=False)
    return {"funnel": funnel, "controls": ctrl, "files": {"funnel_csv": str(run.dir / "funnel.csv"),
                                                          "duo_pairs": str(run.dir / "duo_pairs.csv")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--tics", type=lambda s: [parse_tic(x) for x in s.split(",") if x])
    ap.add_argument("--sector", type=int)
    ap.add_argument("--camera", type=int)
    ap.add_argument("--ccd", type=int)
    ap.add_argument("--control", action="append", choices=sorted(CONTROLS), help="add a known duo as a control")
    ap.add_argument("--max-stars", type=int)
    ap.add_argument("--no-known-filter", action="store_true")
    for k, v in CUTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--snr-min", type=float, default=7.0)
    ap.add_argument("--pmin", type=float, default=10.0)
    ap.add_argument("--veto-tol", type=float, default=0.5)
    ap.add_argument("--veto-max-stars", type=int, default=2)
    ap.add_argument("--redo", action="append", choices=STAGES, default=[])
    ap.add_argument("--no-packets", action="store_true")
    ap.add_argument("--cache-dir", type=Path)
    a = ap.parse_args(argv)
    if not (a.tics or a.control or (a.sector and a.camera and a.ccd)):
        ap.error("give --tics, --control, or --sector with --camera and --ccd")
    cuts = {k: getattr(a, k) for k in CUTS}
    r = run_batch(a.run_dir, a.tics, a.sector, a.camera, a.ccd, cuts, not a.no_known_filter, a.max_stars, a.control,
                  a.snr_min, a.pmin, a.veto_tol, a.veto_max_stars, a.redo, a.cache_dir, not a.no_packets)
    print(json.dumps(jsonable(r), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
