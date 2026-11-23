"""survey: scan many sectors x cameras x CCDs fast. A cheap quick look at every star, the full hunt on the few that show something.

    survey plan --dir surveys/s70-75 --sectors 70-75                 # 6 sectors x 16 CCDs = 96 jobs
    survey run  --dir surveys/s70-75                                 # 150 s chunks (exit 75 = paused; rerun)
    survey run  --dir surveys/s70-75 --detach                        # everything, in the background
    survey status --dir surveys/s70-75                               # one line per job
    survey top  --dir surveys/s70-75 -n 20                           # best candidates across the whole survey

Per CCD job:

  select   TESS-SPOC targets on the CCD, TIC cuts, known TOI/CTOI removed (batch_runner)
  quick    every star: that sector's TESS-SPOC light curve (built from the TIC, no MAST query; plus
           --quick-sectors - 1 nearby earlier sectors), detrend, BLS (pmin 0.5 d) + single-dip search, ~0.4 s.
           Downloads (--dl-threads) overlap with searches (--workers). The file is deleted unless the star
           looks interesting, so disk use stays flat however many CCDs you scan.
  flag     CCD-wide bad windows (many stars dipping together) are found and removed; stars with a periodic
           signal (SNR >= 7, >= 2 transits) or a clean single dip (SNR >= 8) are flagged, at most --max-deep.
  deep     `hunt` on the flagged stars: all sectors, duo funnel, full BLS, triage, gate, vet
           (the CCD's bad windows are passed down).
  classify every lead is given a class and checked against the catalogs (ExoFOP TOI/CTOI, NASA, Prsa EBs,
           Gaia EBs and non-single stars):
             planet_candidate  periodic signal that passed every gate and vet check
             duo_candidate     two single transits years apart, aliases solved, not explained by a periodic signal
             single_transit    one clean deep dip, no repeat in the data (long-period planet, brown dwarf, EB...)
             eb_candidate      strong periodic signal that fails as a planet for EB reasons (odd/even, size, V shape)
           novelty = "new" when no catalog knows the star as a planet host, TOI/CTOI or EB.

Survey outputs: leaderboard.csv (every lead from every CCD, best first), survey_summary.md (short).

What it can and cannot find: the quick tier sees what is visible in --quick-sectors sectors (deep eclipses,
giant planets, strong single dips). Shallow signals that need many sectors to reach SNR 9 (TIC 141618383's
10.43 d signal is SNR ~5 per sector) are missed unless you raise --quick-sectors or run `hunt` on the CCD.
Every lead is a hypothesis for a human; "new" means "not in these catalogs today", not "discovered".
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

from .batch_runner import CUTS, Deferred, Run
from .common import default_cache_dir, jsonable, now_et, today_et, write_json

EXIT_PAUSED = 75
QUICK = {"snr_periodic": 7.0, "min_transits": 2, "snr_single": 8.0, "keep_snr": 6.0}
CLASS_ORDER = {"planet_candidate": 0, "duo_candidate": 1, "single_transit": 2, "eb_candidate": 3}
LB_COLS = ["rank", "class", "novelty", "tic", "sector", "camera", "ccd", "period_d", "epoch_btjd", "duration_h",
           "depth_ppm", "snr", "why", "catalog_flags", "evidence"]
SIGNAL_KEYS = ["period_d", "epoch_btjd", "duration_h", "depth_ppm", "snr", "n_transits", "odd_even_sigma",
               "harmonic_flag", "duration_at_grid_edge", "sinusoid_like"]


# ---------------------------------------------------------------- plan
def parse_ranges(s):
    out = []
    for part in str(s).split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def plan(d, sectors, cameras=(1, 2, 3, 4), ccds=(1, 2, 3, 4)):
    d = Path(d)
    p = d / "plan.json"
    old = json.loads(p.read_text()) if p.exists() else {"jobs": []}
    have = {(j["sector"], j["camera"], j["ccd"]) for j in old["jobs"]}
    jobs = old["jobs"] + [{"sector": s, "camera": c, "ccd": k, "status": "pending"}
                          for s in sectors for c in cameras for k in ccds if (s, c, k) not in have]
    write_json(p, {"created_at": old.get("created_at", now_et()), "jobs": jobs})
    return jobs


def job_name(j):
    return f"s{j['sector']}_c{j['camera']}_d{j['ccd']}"


# ---------------------------------------------------------------- quick tier
def spoc_uri(tic, sector):
    t16 = f"{int(tic):016d}"
    sub = "/".join(t16[i:i + 4] for i in range(0, 16, 4))
    return (f"mast:HLSP/tess-spoc/s{sector:04d}/target/{sub}/hlsp_tess-spoc_tess_phot_{t16}-s{sector:04d}_tess_v1_lc.fits")


def lc_dest(cache_dir, tic, sector):
    return Path(cache_dir) / "lc" / f"TIC{int(tic)}" / spoc_uri(tic, sector).rsplit("/", 1)[1]


def _download(cache_dir, tic, sectors):
    """Files for these sectors (already cached ones reused). Returns (files, downloaded_now, misses)."""
    from .tess_lc_fetch import _download as dl, _is_good_fits

    files, new, miss = [], [], []
    for s in sectors:
        dest = lc_dest(cache_dir, tic, s)
        if dest.exists() and _is_good_fits(dest):
            files.append(str(dest))
            continue
        try:
            dl(spoc_uri(tic, s), dest)
            files.append(str(dest))
            new.append(str(dest))
        except Exception as exc:  # noqa: BLE001 - not a TESS-SPOC target that sector, or archive trouble
            miss.append(f"s{s}: {str(exc)[:120]}")
    return files, new, miss


def _quick_search(run_dir, cache_dir, tic, files, new_files, misses, keep_snr):
    """Search one star's quick light curves; delete the files we downloaded unless something is there."""
    from .allsector_gate import read_lc_file
    from .transit_search import search

    run = Run(run_dir, cache_dir=cache_dir)

    def work():
        if not files:
            return {"ok": False, "tic": tic, "error": "no light curve: " + "; ".join(misses)[:300]}
        crowd = [s.get("crowdsap") for f in files for s in read_lc_file(f) if s.get("crowdsap") is not None]
        r = search(tic, lc_files=files, cache_dir=cache_dir, pmin=0.5, pmax=None, n_signals=1, bin_min=10.0,
                   out_dir=run.dir / "quick" / f"TIC{tic}", write_csv=False)
        per = [{k: s.get(k) for k in SIGNAL_KEYS} for s in r["signals"] if s["kind"] == "periodic"][:1]
        singles = sorted([{"epoch_btjd": s["epoch_btjd"], "duration_h": s["duration_h"], "depth_ppm": s["depth_ppm"],
                           "snr": s["snr"], "sector": s["sector"], "near_gap_or_edge": s["near_gap_or_edge"]}
                          for s in r["signals"] if s["kind"] == "single"], key=lambda x: -x["snr"])[:8]
        return {"ok": True, "tic": tic, "sectors": r["inputs"]["sectors"], "crowdsap_min": min(crowd) if crowd else None,
                "noise_ppm": r["noise_ppm"], "coverage": r["coverage"], "periodic": per, "singles": singles,
                "files": files}
    out = run.cached("quick", f"TIC{tic}", work)
    best = max([p["snr"] or 0 for p in out.get("periodic", [])] + [s["snr"] for s in out.get("singles", [])
                                                                    if not s["near_gap_or_edge"]] + [0])
    keep = out.get("ok") and best >= keep_snr
    if not keep:
        for f in new_files:
            Path(f).unlink(missing_ok=True)
    return bool(keep)


def quick_sectors_for(stars, sector, n):
    """{tic: [sector, earlier sectors nearest in time...]} from tess-point (no archive query)."""
    if n <= 1:
        return {t: [sector] for t in stars}
    from tess_stars2px import tess_stars2px_function_entry

    ids = [t for t in stars if stars[t].get("ra") is not None]
    out = tess_stars2px_function_entry(ids, [stars[t]["ra"] for t in ids], [stars[t]["dec"] for t in ids])
    seen = {}
    for i, s in zip(out[0], out[3]):
        if int(s) < sector:
            seen.setdefault(int(i), set()).add(int(s))
    return {t: [sector] + sorted(seen.get(t, ()), reverse=True)[:n - 1] for t in stars}


def quick_tier(a, run, stars, deadline, log):
    sec_map = quick_sectors_for(stars, run.sector, a.quick_sectors)
    todo = [t for t in sorted(stars) if not run.path("quick", f"TIC{t}").exists()]
    total, done = len(stars), len(stars) - len(todo)
    if not todo:
        return True
    import multiprocessing as mp

    cpu = (ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) if a.workers > 1
           else ThreadPoolExecutor(max_workers=1))
    it = iter(todo)
    dls, srch = {}, set()
    with ThreadPoolExecutor(max_workers=a.dl_threads) as io_ex, cpu:
        while True:
            while len(dls) < 2 * a.dl_threads and len(srch) < 4 * a.workers and (deadline is None or time.time() < deadline):
                t = next(it, None)
                if t is None:
                    break
                dls[io_ex.submit(_download, a.cache_dir, t, sec_map[t])] = t
            if not dls and not srch:
                break
            fin, _ = wait(set(dls) | srch, return_when=FIRST_COMPLETED)
            for f in fin:
                if f in dls:
                    t = dls.pop(f)
                    try:
                        files, new, miss = f.result()
                    except Exception as exc:  # noqa: BLE001
                        files, new, miss = [], [], [str(exc)[:200]]
                    srch.add(cpu.submit(_quick_search, run.dir, a.cache_dir, t, files, new, miss, QUICK["keep_snr"]))
                else:
                    srch.discard(f)
                    try:
                        f.result()
                    except Exception as exc:  # noqa: BLE001
                        log(f"quick crash: {type(exc).__name__}: {exc}"[:200])
                    done += 1
                    if done % 100 == 0:
                        log(f"quick {done}/{total}")
    return done == total


def flag(a, run):
    """CCD bad windows, then the stars worth a deep look."""
    from .bls_triage import bad_windows, frac_in_bad, transit_times

    q = [d for d in load_stage(run.dir, "quick") if d.get("ok")]
    wins, _ = bad_windows([{"tic": d["tic"], "singles": d["singles"]} for d in q])
    rows = []
    for d in q:
        if d.get("crowdsap_min") is not None and d["crowdsap_min"] < a.cuts["crowdsap_min"]:
            continue
        why, score = [], 0.0
        for p in d["periodic"]:
            P = p["period_d"]
            if ((p["snr"] or 0) >= QUICK["snr_periodic"] and (p["n_transits"] or 0) >= QUICK["min_transits"]
                    and not p["harmonic_flag"] and not (P >= 0.9 and abs(P - round(P)) < 0.03)
                    and frac_in_bad(transit_times(p, d["coverage"]), wins) < 1 / 3):
                why.append(f"periodic P={P:.4f} snr={p['snr']:.1f} depth={p['depth_ppm']:.0f}")
                score = max(score, p["snr"])
        for s in d["singles"]:
            if s["snr"] >= QUICK["snr_single"] and not s["near_gap_or_edge"] and \
                    not any(w["start_btjd"] <= s["epoch_btjd"] <= w["end_btjd"] for w in wins):
                why.append(f"single t={s['epoch_btjd']:.2f} snr={s['snr']:.1f} depth={s['depth_ppm']:.0f}")
                score = max(score, s["snr"])
                break
        if why:
            rows.append({"tic": d["tic"], "score": round(score, 2), "why": why})
    rows.sort(key=lambda r: -r["score"])
    res = {"n_quick_ok": len(q), "n_flagged": len(rows), "n_deep": min(len(rows), a.max_deep), "bad_windows": wins,
           "flagged": rows[:a.max_deep], "n_variable": sum(1 for d in q for p in d["periodic"]
                                                            if p.get("sinusoid_like") and (p["snr"] or 0) >= 20)}
    write_json(run.dir / "flagged.json", res)
    return res


# ---------------------------------------------------------------- deep tier
def deep(a, j, run, flagged, deadline, log):
    from . import hunt

    tics = [r["tic"] for r in flagged["flagged"]]
    if not tics:
        return 0
    dd = run.dir / "deep"
    dd.mkdir(parents=True, exist_ok=True)
    write_json(dd / "bad_windows_extra.json", flagged["bad_windows"])
    rem = 0 if deadline is None else max(1.0, deadline - time.time())
    argv = ["run", "--run-dir", str(dd), "--tics", ",".join(map(str, tics)), "--sector", str(j["sector"]),
            "--camera", str(j["camera"]), "--ccd", str(j["ccd"]), "--budget", str(rem), "--workers", str(a.workers),
            "--dl-threads", str(a.dl_threads), "--cache-dir", str(a.cache_dir), "--tpf-sectors", str(a.tpf_sectors)]
    for k in CUTS:
        argv += ["--" + k.replace("_", "-"), str(a.cuts[k])]
    argv += (["--no-controls"] if a.no_controls else []) + (["--no-known-filter"] if a.no_known_filter else [])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = hunt.main(argv)
    with (run.dir / "deep.log").open("a") as fh:
        fh.write(f"[{now_et()}] exit {code}\n{buf.getvalue()}")
    if code == 2:
        raise RuntimeError("control failed in deep tier: " + buf.getvalue()[-300:])
    return code


# ---------------------------------------------------------------- classify
def _novelty(v):
    f = v.get("flags") or {}
    if any(f.get(k) for k in ("known_planet", "toi", "ctoi")):
        return "known_toi_or_planet"
    if f.get("eb") or f.get("period_matches_catalog"):
        return "known_eb_or_period"
    if v.get("summary", {}).get("sources_down"):
        return "unchecked (catalog down)"
    if f.get("nss_binary"):
        return "new_eclipses_known_binary"
    if f.get("nearby_eb"):
        return "new (EB nearby)"
    return "new"


def classify(a, j, run, deadline):
    from . import hunt

    dd = run.dir / "deep"
    rows = []
    base = {"sector": j["sector"], "camera": j["camera"], "ccd": j["ccd"]}
    gates = {(g["tic"], round(g["period_d"], 4)): g for g in load_stage(dd, "gate")}
    tri = {}
    if (dd / "triage.json").exists():
        t = json.loads((dd / "triage.json").read_text())
        tri = {(r["tic"], round(r["period_d"], 4)): r for r in t["survivors"] + t["near_misses"]}
    for v in load_stage(dd, "vet"):
        if v.get("status") == "gate_passed_needs_human_review":
            k = (v["tic"], round(v["period_d"], 4))
            g, tr = gates.get(k, {}), tri.get(k, {})
            rows.append({**base, "class": "planet_candidate", "tic": v["tic"], "period_d": v["period_d"],
                         "epoch_btjd": g.get("epoch_btjd"), "duration_h": g.get("duration_h"),
                         "depth_ppm": g.get("reference_depth_ppm") or g.get("depth_ppm"), "snr": tr.get("snr"),
                         "novelty": "new" if v.get("catalog_veto") == "clean" else str(v.get("catalog_veto")),
                         "catalog_flags": ",".join(v.get("catalog_flags") or []), "evidence": v.get("packet_dir"),
                         "why": "passed gate, catalog, secondary and centroid checks"})
    fun = json.loads((dd / "funnel.json").read_text()) if (dd / "funnel.json").exists() else {}
    duo_tics = set()
    for p in fun.get("packets", []):
        r = next((x for x in fun.get("duo", []) if x["tic"] == p["tic"] and abs(x["t1"] - p["t1"]) < 1e-3), {})
        duo_tics.add(int(p["tic"]))
        if p.get("status") == "duo_candidate_needs_human_review" and not hunt._bls_explains(
                dd, p["tic"], p["t1"], p["t2"], r.get("duration_h", 6)):
            rows.append({**base, "class": "duo_candidate", "tic": p["tic"], "epoch_btjd": p["t1"],
                         "depth_ppm": r.get("depth_ppm"), "duration_h": r.get("duration_h"), "snr": r.get("snr"),
                         "novelty": "new" if p.get("catalog") == "clean" else str(p.get("catalog")),
                         "evidence": p.get("packet_dir"),
                         "why": f"t1={p['t1']:.2f} t2={p['t2']:.2f}; P in {str(r.get('surviving_periods'))[:60]}"})
    # EB-like periodic signals: strong, and failing as a planet for eclipse reasons. Spotted / pulsating stars
    # also fail (wide sinusoid-like boxes, T14 far too long for the star), so those count as variables instead.
    ebs, n_var = [], 0
    if (dd / "triage.csv").exists():
        with (dd / "triage.csv").open() as fh:
            for r in csv.DictReader(fh):
                fails = set(filter(None, r["fails"].split(";")))
                snr = float(r["snr"] or 0)
                if snr < a.eb_snr:
                    continue
                tic, P = int(r["tic"]), float(r["period_d"])
                g = gates.get((tic, round(P, 4)), {})
                gfail = {k for k, v in (g.get("tests") or {}).items() if v == "fail"}
                bp = dd / "stages" / "bls" / f"TIC{tic}.json"
                top = next((x for x in (json.loads(bp.read_text()).get("signals", []) if bp.exists() else [])
                            if abs(x["period_d"] - P) < 1e-4), {})
                ratio = float(r["t14_ratio"]) if r.get("t14_ratio") not in (None, "") else None
                too_long = ratio is not None and ratio > 1.5
                grazing = "duration" in fails and ratio is not None and ratio < 1.0
                shape_ok = not top.get("sinusoid_like") and not too_long
                why = set()
                if "secondary" in gfail:
                    why.add("secondary eclipse")
                if grazing or ("duration_density" in gfail and not too_long):
                    why.add("grazing/V-shaped (T14 too short)")
                if shape_ok and ("odd_even" in fails or "odd_even" in gfail):
                    why.add("odd/even depths differ")
                if shape_ok and "radius" in fails and float(r["depth_ppm"] or 0) >= 5000:
                    why.add("too large for a planet")
                if why:
                    ebs.append({**base, "class": "eb_candidate", "tic": tic, "period_d": P,
                                "epoch_btjd": float(r["epoch_btjd"]), "duration_h": float(r["duration_h"]),
                                "depth_ppm": float(r["depth_ppm"] or 0), "snr": snr, "why": "; ".join(sorted(why)),
                                "evidence": str(dd / "triage.csv")})
                elif too_long or top.get("sinusoid_like"):
                    n_var += 1
    ebs.sort(key=lambda r: -r["snr"])
    # strong single dips that nothing else explains
    singles = []
    flagged = json.loads((run.dir / "flagged.json").read_text()) if (run.dir / "flagged.json").exists() else {"flagged": []}
    wins = flagged.get("bad_windows", [])
    for f in flagged["flagged"]:
        q = json.loads(run.path("quick", f"TIC{f['tic']}").read_text())
        for s in q.get("singles", []):
            if s["snr"] < a.single_snr or s["near_gap_or_edge"] or int(f["tic"]) in duo_tics or \
                    any(w["start_btjd"] <= s["epoch_btjd"] <= w["end_btjd"] for w in wins):
                continue
            if hunt._bls_explains(dd, f["tic"], s["epoch_btjd"], s["epoch_btjd"], s["duration_h"], snr_min=7.0):
                continue
            singles.append({**base, "class": "single_transit", "tic": int(f["tic"]), "epoch_btjd": s["epoch_btjd"],
                            "duration_h": s["duration_h"], "depth_ppm": s["depth_ppm"], "snr": s["snr"],
                            "why": f"one dip in S{s['sector']}, no repeat found in the data",
                            "evidence": str(run.path("quick", f"TIC{f['tic']}"))})
            break
    singles.sort(key=lambda r: -r["snr"])
    # catalog novelty for the strongest EB and single leads (network; cached per star)
    from .catalog_veto import veto

    for r in ebs[:a.max_veto] + singles[:a.max_veto]:
        key = f"veto_TIC{r['tic']}_{r.get('period_d') or 0:.4f}"
        v = run.cached("classify", key, lambda r=r: {"ok": True, **veto(
            r["tic"], period_d=r.get("period_d"), epoch_btjd=r.get("epoch_btjd"), depth_ppm=r.get("depth_ppm"),
            cache_dir=a.cache_dir)})
        r["novelty"] = _novelty(v) if v.get("ok") else "unchecked (error)"
        r["catalog_flags"] = ",".join(k for k, x in (v.get("flags") or {}).items() if x)
    for r in ebs[a.max_veto:] + singles[a.max_veto:]:
        r["novelty"] = "unchecked (below --max-veto)"
    rows += ebs + singles
    out = {"job": job_name(j), "rows": rows, "counts": {c: sum(r["class"] == c for r in rows) for c in CLASS_ORDER},
           "n_variable": (flagged.get("n_variable") or 0) + n_var}
    write_json(run.dir / "classify.json", out)
    return out


# ---------------------------------------------------------------- survey loop
def load_stage(run_dir, stage):
    return [json.loads(p.read_text()) for p in sorted((Path(run_dir) / "stages" / stage).glob("*.json"))]


def rank_key(r):
    return (CLASS_ORDER.get(r["class"], 9), 0 if str(r.get("novelty", "")).startswith("new") else 1, -(r.get("snr") or 0))


def leaderboard(d):
    d = Path(d)
    rows = []
    for p in sorted((d / "jobs").glob("*/classify.json")):
        rows += json.loads(p.read_text())["rows"]
    rows.sort(key=rank_key)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    with (d / "leaderboard.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LB_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(jsonable(rows))
    return rows


def _f(x, nd):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{float(x):.{nd}f}"


def write_summary(d, jobs):
    d = Path(d)
    rows = leaderboard(d)
    done = [j for j in jobs if j["status"] == "done"]
    n_stars = sum(j.get("n_stars", 0) or 0 for j in jobs)
    n_quick = sum(j.get("n_quick_ok", 0) or 0 for j in jobs)
    cnt = {c: sum(r["class"] == c for r in rows) for c in CLASS_ORDER}
    new = {c: sum(r["class"] == c and str(r.get("novelty", "")).startswith("new") for r in rows) for c in CLASS_ORDER}
    L = [f"# Survey summary ({today_et()})", "",
         f"**{len(done)}/{len(jobs)} CCDs done; {n_quick} of {n_stars} selected stars quick-searched; "
         + ", ".join(f"{c} {cnt[c]} ({new[c]} new)" for c in CLASS_ORDER) + "**", "",
         "| CCD | status | stars | quick ok | flagged | deep | planet | duo | single | EB | variables |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for j in jobs:
        c = j.get("counts") or {}
        L.append(f"| {job_name(j)} | {j['status']} | {j.get('n_stars', '')} | {j.get('n_quick_ok', '')} | "
                 f"{j.get('n_flagged', '')} | {j.get('n_deep', '')} | {c.get('planet_candidate', '')} | "
                 f"{c.get('duo_candidate', '')} | {c.get('single_transit', '')} | {c.get('eb_candidate', '')} | {j.get('n_variable', '')} |")
    if rows:
        L += ["", "## Top leads", "", "| # | class | novelty | TIC | CCD | P (d) | depth ppm | SNR | why |", "|---|---|---|---|---|---|---|---|---|"]
        for r in rows[:25]:
            L.append(f"| {r['rank']} | {r['class']} | {r.get('novelty', '')} | {r['tic']} | s{r['sector']}/{r['camera']}/{r['ccd']} | "
                     f"{_f(r.get('period_d'), 4)} | {_f(r.get('depth_ppm'), 0)} | {_f(r.get('snr'), 1)} | {str(r.get('why', ''))[:70]} |")
    L += ["", f"Full list: `{d / 'leaderboard.csv'}`. Per-CCD detail: `{d}/jobs/<ccd>/deep/summary.md`.", "",
          "Every lead is a hypothesis. \"new\" means not in ExoFOP TOI/CTOI, NASA, Prsa+2022 EBs or Gaia DR3 EB/NSS today. "
          "Shallow multi-sector signals are outside the quick tier's reach (see docs/survey.md)."]
    (d / "survey_summary.md").write_text("\n".join(L) + "\n")
    return rows


def run_survey(a) -> int:
    d = Path(a.dir)
    t0 = time.time()
    deadline = t0 + a.budget if a.budget else None
    pj = json.loads((d / "plan.json").read_text())
    jobs = pj["jobs"]
    logf = (d / "survey.log").open("a")

    def log(msg):
        logf.write(f"[{now_et()}] {msg}\n")
        logf.flush()

    def save():
        write_json(d / "plan.json", pj)

    paused = None
    try:
        for j in jobs:
            if j["status"] == "done" or (j["status"] == "failed" and j.get("attempts", 0) >= a.max_attempts):
                continue
            if j["status"] == "failed":   # transient archive trouble is common: clear cached errors and retry
                for st in ("select", "quick", "classify"):
                    for p in (d / "jobs" / job_name(j) / "stages" / st).glob("*.json"):
                        if json.loads(p.read_text()).get("ok") is False:
                            p.unlink()
                j["status"] = "pending"
                j.pop("error", None)
            if deadline is not None and time.time() > deadline:
                raise Deferred(job_name(j))
            run = Run(d / "jobs" / job_name(j), cache_dir=a.cache_dir, deadline=deadline)
            run.sector = j["sector"]
            try:
                from .batch_runner import stage_select

                j["status"] = "select"
                save()
                sel = stage_select(run, None, j["sector"], j["camera"], j["ccd"], a.cuts, not a.no_known_filter,
                                   a.max_stars, [])
                if not sel.get("ok"):
                    raise RuntimeError(f"select: {sel.get('error')}")
                stars = {int(s["tic"]): s for s in sel["selected"]}
                j.update(status="quick", n_stars=len(stars), started_at=j.get("started_at") or now_et())
                save()
                log(f"{job_name(j)}: {len(stars)} stars")
                if not quick_tier(a, run, stars, deadline, log):
                    raise Deferred(f"{job_name(j)} quick")
                fl = flag(a, run)
                j.update(status="deep", n_quick_ok=fl["n_quick_ok"], n_flagged=fl["n_flagged"], n_deep=fl["n_deep"])
                save()
                if not a.no_deep:
                    code = deep(a, j, run, fl, deadline, log)
                    if code == EXIT_PAUSED:
                        raise Deferred(f"{job_name(j)} deep")
                    if code not in (0, EXIT_PAUSED):
                        log(f"{job_name(j)}: deep exit {code}")
                c = classify(a, j, run, deadline)
                j.update(status="done", counts=c["counts"], n_variable=c["n_variable"], finished_at=now_et())
                save()
                write_summary(d, jobs)
                log(f"{job_name(j)}: done {c['counts']}")
            except Deferred:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad CCD must not stop the survey
                j.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:400], attempts=j.get("attempts", 0) + 1)
                save()
                log(f"{job_name(j)}: FAILED {j['error']}")
                if "control failed" in str(exc):
                    print(f"CONTROL FAILED in {job_name(j)}; survey stopped. See {d / 'survey.log'}")
                    return 2
    except Deferred as df:
        paused = str(df)
    save()
    rows = write_summary(d, jobs)
    el = time.time() - t0
    done = sum(j["status"] == "done" for j in jobs)
    head = f"{done}/{len(jobs)} CCDs done, {len(rows)} leads ({sum(str(r.get('novelty', '')).startswith('new') for r in rows)} new)"
    if paused:
        print(f"PAUSED at {paused} after {el:.0f}s; {head}. Re-run the same command to continue.")
        return EXIT_PAUSED
    print(f"DONE in {el:.0f}s; {head}. {d / 'survey_summary.md'}")
    return 0


def status(d) -> int:
    d = Path(d)
    pj = json.loads((d / "plan.json").read_text())
    pid = d / "survey.pid"
    alive = False
    if pid.exists():
        try:
            os.kill(int(pid.read_text()), 0)
            alive = True
        except (OSError, ValueError):
            pass
    jobs = pj["jobs"]
    by = {}
    for j in jobs:
        by[j["status"]] = by.get(j["status"], 0) + 1
    print(f"{'running' if alive else 'not running'}; " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
    for j in jobs:
        if j["status"] not in ("pending",):
            c = j.get("counts") or {}
            print(f"  {job_name(j):12s} {j['status']:6s} stars {j.get('n_stars', '-')} flagged {j.get('n_flagged', '-')} "
                  + " ".join(f"{k.split('_')[0]}={v}" for k, v in c.items() if v) + (f" ERR {j['error'][:80]}" if j.get("error") else ""))
    return 0


def top(d, n) -> int:
    rows = leaderboard(d)
    for r in rows[:n]:
        p = f"P={r['period_d']:.4f}" if r.get("period_d") else f"t={r.get('epoch_btjd', 0):.2f}"
        print(f"{r['rank']:3d} {r['class']:16s} {str(r.get('novelty', '')):24s} TIC{r['tic']} s{r['sector']}/{r['camera']}/{r['ccd']} "
              f"{p} depth={r.get('depth_ppm') or 0:.0f} snr={r.get('snr') or 0:.1f} {str(r.get('why', ''))[:50]}")
    if not rows:
        print("no leads yet")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--sectors", required=True, help="e.g. 70-75 or 61,63,70-72")
    p.add_argument("--cameras", default="1-4")
    p.add_argument("--ccds", default="1-4")
    r = sub.add_parser("run")
    r.add_argument("--dir", type=Path, required=True)
    r.add_argument("--budget", type=float, default=150.0, help="seconds before pausing (0 = no limit)")
    r.add_argument("--detach", action="store_true")
    r.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 2))
    r.add_argument("--dl-threads", type=int, default=6, help="parallel downloads (MAST throttles at ~12)")
    r.add_argument("--quick-sectors", type=int, default=1, help="sectors per star in the quick tier (more = deeper, slower)")
    r.add_argument("--max-deep", type=int, default=150, help="flagged stars per CCD sent to the deep tier")
    r.add_argument("--max-stars", type=int, help="stars per CCD (for trial runs)")
    r.add_argument("--max-veto", type=int, default=15, help="EB and single leads per CCD checked against the catalogs")
    r.add_argument("--eb-snr", type=float, default=12.0)
    r.add_argument("--single-snr", type=float, default=10.0)
    r.add_argument("--tpf-sectors", type=int, default=4)
    r.add_argument("--no-deep", action="store_true", help="quick tier + flagging only")
    r.add_argument("--max-attempts", type=int, default=3, help="retries for a CCD that failed (archive trouble)")
    r.add_argument("--no-controls", action="store_true")
    r.add_argument("--no-known-filter", action="store_true")
    for k, v in CUTS.items():
        r.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    r.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    for name in ("status", "summary"):
        s = sub.add_parser(name)
        s.add_argument("--dir", type=Path, required=True)
    rc = sub.add_parser("reclassify", help="re-apply the classify rules to finished CCDs (catalog checks are cached)")
    rc.add_argument("--dir", type=Path, required=True)
    for k, v in (("max_veto", 15), ("eb_snr", 12.0), ("single_snr", 10.0)):
        rc.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    rc.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    t = sub.add_parser("top")
    t.add_argument("--dir", type=Path, required=True)
    t.add_argument("-n", type=int, default=20)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        jobs = plan(a.dir, parse_ranges(a.sectors), parse_ranges(a.cameras), parse_ranges(a.ccds))
        print(f"{len(jobs)} CCD jobs in {a.dir / 'plan.json'}")
        return 0
    if a.cmd == "status":
        return status(a.dir)
    if a.cmd == "summary":
        print((a.dir / "survey_summary.md").read_text())
        return 0
    if a.cmd == "top":
        return top(a.dir, a.n)
    if a.cmd == "reclassify":
        pj = json.loads((a.dir / "plan.json").read_text())
        for j in pj["jobs"]:
            if j["status"] == "done":
                c = classify(a, j, Run(a.dir / "jobs" / job_name(j), cache_dir=a.cache_dir), None)
                j.update(counts=c["counts"], n_variable=c["n_variable"])
                j.pop("error", None)
        write_json(a.dir / "plan.json", pj)
        write_summary(a.dir, pj["jobs"])
        return top(a.dir, 20)
    if a.detach:
        args = [x for x in argv if x != "--detach"]
        if "--budget" in args:
            i = args.index("--budget")
            del args[i:i + 2]
        log = open(a.dir / "survey.out", "ab")
        pr = subprocess.Popen([sys.executable, "-m", "shared_tools.exo.survey", *args, "--budget", "0"], stdout=log,
                              stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        (a.dir / "survey.pid").write_text(str(pr.pid))
        print(f"started pid {pr.pid}; check with: survey status --dir {a.dir}")
        return 0
    a.cuts = {k: getattr(a, k) for k in CUTS}
    (a.dir / "survey.pid").write_text(str(os.getpid()))
    return run_survey(a)


if __name__ == "__main__":
    raise SystemExit(main())
