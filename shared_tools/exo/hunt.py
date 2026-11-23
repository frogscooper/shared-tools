"""hunt: one resumable command for a whole TESS CCD (or a TIC list), duo + BLS, with a short summary.

    hunt run --sector 74 --camera 1 --ccd 2 --run-dir runs/s74_c1_d2            # time-boxed chunks (default 150 s)
    hunt run --sector 74 --camera 1 --ccd 2 --run-dir runs/s74_c1_d2 --detach   # whole thing in the background
    hunt status --run-dir runs/s74_c1_d2                                        # one line per stage

Stages (everything is cached under <run>/stages/<stage>/, so a rerun continues where the last one stopped):

  controls  TOI-4600 c through the duo funnel and pi Men c through BLS. Cached per version of the tool code,
            so they run once, not once per CCD. The hunt stops if a control fails.
  select    TESS-SPOC targets on the CCD, TIC cuts, known TOI/CTOI removed (batch_runner)
  fetch     coverage + light curves, --dl-threads parallel downloads (MAST 429s back off and retry)
  dips      single-dip search per star, --workers processes (feeds the duo funnel)
  duo       batch_runner duo funnel, one pass with the in-sector rule dropped and each pair labelled in_sector;
            dip fragments merged, stars shown periodic demoted, duo packets for what is left
  bls       transit_search per star (pmin 0.5, pmax 60, 2 signals), --workers processes
  triage    bls_triage cuts (incl. duration vs density, shared-epoch systematics)
  gate      allsector_gate (+ duration_density) on survivors and near misses
  vet       candidate_packet (catalog incl. Gaia NSS, secondary, centroid) for gate passes
  summary   <run>/summary.md (short) and <run>/hunt-log-draft.md (project-log format)

With --budget N (seconds) no new work starts after N s; in-flight items finish, progress is saved and the
exit code is 75. Run the same command again to continue. --budget 0 means no limit (used by --detach).
Nothing is submitted anywhere. A "candidate for review" is a hypothesis for a human.
"""
from __future__ import annotations

import argparse
import hashlib
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

STAGES = ["controls", "select", "fetch", "dips", "duo", "bls", "triage", "gate", "vet", "summary"]
EXIT_PAUSED = 75
BLS = {"pmin": 0.5, "pmax": 60.0, "n_signals": 2}
PI_MEN = {"tic": 261136679, "period_d": 6.26782, "sectors": [1, 4, 8, 11, 12, 13], "name": "pi Men c"}
SIGNAL_KEYS = ["period_d", "epoch_btjd", "duration_h", "depth_ppm", "depth_err_ppm", "snr", "sde", "n_transits",
               "odd_even_sigma", "harmonic_flag", "harmonic_of", "duration_at_grid_edge", "sinusoid_like", "n_seasons"]


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        if p.name != "hunt.py":      # orchestration changes do not change what the controls test
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- progress
class Progress:
    def __init__(self, run_dir):
        self.path = Path(run_dir) / "progress.json"
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {"stages": {}}
        self.d.update(pid=os.getpid(), host_started_at=now_et())
        self._t = 0.0

    def set(self, stage, force=True, **kw):
        self.d["stages"].setdefault(stage, {}).update(kw, updated_at=now_et())
        if force or time.time() - self._t > 5:
            self.d["updated_at"] = now_et()
            write_json(self.path, self.d)
            self._t = time.time()


# ---------------------------------------------------------------- workers (top level: picklable)
def _run(run_dir, cache_dir):
    return Run(run_dir, cache_dir=cache_dir)


def _fetch_job(run_dir, cache_dir, tic, cuts):
    from .batch_runner import stage_coverage, stage_lightcurves

    run = _run(run_dir, cache_dir)
    stage_coverage(run, tic, cuts)
    return stage_lightcurves(run, tic, cuts).get("ok")


def _dips_job(run_dir, cache_dir, tic, files, snr_min):
    from .batch_runner import stage_dips

    return stage_dips(_run(run_dir, cache_dir), tic, files, snr_min).get("ok")


def _bls_job(run_dir, cache_dir, tic, files, rstar, rho, pmin, pmax, n_signals, grid_oversample):
    from .transit_search import search

    run = _run(run_dir, cache_dir)

    def work():
        r = search(tic, lc_files=files, cache_dir=cache_dir, pmin=pmin, pmax=pmax, n_signals=n_signals,
                   out_dir=run.dir / "bls" / f"TIC{tic}", grid_oversample=grid_oversample, write_csv=False)
        per = [{k: s.get(k) for k in SIGNAL_KEYS} | {"kind": "periodic"} for s in r["signals"] if s["kind"] == "periodic"]
        singles = [{"epoch_btjd": s["epoch_btjd"], "duration_h": s["duration_h"], "depth_ppm": s["depth_ppm"],
                    "snr": s["snr"], "sector": s["sector"]} for s in r["signals"] + r["singles_matching_periodic"]
                   if s["kind"] == "single"]
        return {"ok": True, "tic": tic, "rstar": rstar, "rho_solar": rho, "sectors": r["inputs"]["sectors"],
                "noise_ppm": r["noise_ppm"], "coverage": r["coverage"], "signals": per, "singles": singles,
                "n_rejected_periodic": len(r["rejected_periodic"]), "json": r["json"]}
    out = run.cached("bls", f"TIC{tic}", work)
    return out.get("ok")


def _gate_job(run_dir, cache_dir, key, tic, files, P, T0, dur_h, depth, rho):
    from .allsector_gate import gate

    run = _run(run_dir, cache_dir)

    def work():
        g = gate(tic, P, T0, dur_h, depth, None, cache_dir, files, run.dir / "gate" / key, rho_solar=rho)
        return {"ok": True, "tic": tic, "period_d": P, "epoch_btjd": T0, "duration_h": dur_h, "depth_ppm": depth,
                "verdict": g["verdict"], "reasons": [str(x) for x in g["reasons"]][:4],
                "tests": {k: v["result"] for k, v in g["tests"].items()}, "n_clean": g["n_clean"], "n_events": g["n_events"],
                "reference_depth_ppm": g["reference_depth_ppm"], "duration": g["tests"].get("duration_density"),
                "plot": g.get("plot"), "json": g["json"]}
    return run.cached("gate", key, work).get("ok")


def pool(run, stage, jobs, fn, workers, kind, deadline, prog, label=None):
    """Run fn(**kw) for (key, kw) jobs whose stage file is missing. Returns True when every job is done."""
    todo = [(k, kw) for k, kw in jobs if not run.path(stage, k).exists()]
    done, total = len(jobs) - len(todo), len(jobs)
    stage = label or stage   # progress name (the stage files keep their batch_runner name)
    prog.set(stage, done=done, total=total, status="running" if todo else "done")
    if not todo:
        return True
    if kind == "process" and workers > 1:
        import multiprocessing as mp

        # spawn, not fork: numpy/wotan load OpenMP, and forking an OpenMP process kills the child
        ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    else:
        ex = ThreadPoolExecutor(max_workers=max(1, workers))
    it, inflight = iter(todo), set()
    with ex:
        while True:
            while len(inflight) < 2 * max(1, workers) and (deadline is None or time.time() < deadline):
                j = next(it, None)
                if j is None:
                    break
                inflight.add(ex.submit(fn, **j[1]))
            if not inflight:
                break
            fin, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for f in fin:
                try:
                    f.result()
                except Exception as exc:  # noqa: BLE001 - recorded by Run.cached in the worker; this is a crash
                    prog.d.setdefault("crashes", []).append(f"{stage}: {type(exc).__name__}: {exc}"[:300])
                    continue
                done += 1
            prog.set(stage, force=False, done=done, total=total)
    complete = done == total
    prog.set(stage, done=done, total=total, status="done" if complete else "paused")
    return complete


# ---------------------------------------------------------------- stages
def stage_controls(a, prog, deadline):
    from .batch_runner import run_batch
    from .transit_search import search

    ch = code_hash()
    root = Path(a.cache_dir) / "hunt_controls" / ch
    run = Run(root, cache_dir=a.cache_dir, deadline=deadline)
    prog.set("controls", status="running", code_hash=ch)

    def duo():
        r = run_batch(root / "toi4600c", control=["toi4600c"], packets=False, cache_dir=a.cache_dir, deadline=deadline)
        c = r["controls"][0]
        return {"ok": bool(c["pair_found"] and c["true_period_survives"] and c["duo_gate"] == "pass"), **c}

    def bls():
        r = search(PI_MEN["tic"], sectors=PI_MEN["sectors"], cache_dir=a.cache_dir, pmin=BLS["pmin"], pmax=30.0,
                   n_signals=1, out_dir=root / "pimen", write_csv=False)
        top = next((s for s in r["signals"] if s["kind"] == "periodic"), {})
        ok = abs((top.get("period_d") or 0) - PI_MEN["period_d"]) < 0.002
        return {"ok": ok, "control": PI_MEN["name"], "period_d": top.get("period_d"), "snr": top.get("snr"),
                "odd_even_sigma": top.get("odd_even_sigma"), "sectors": r["inputs"]["sectors"]}
    out = {"duo": run.cached("controls", "duo_toi4600c", duo), "bls": run.cached("controls", "bls_pimen", bls)}
    ok = all(v.get("ok") for v in out.values())
    prog.set("controls", status="done" if ok else "failed", result={k: {kk: v.get(kk) for kk in
             ("ok", "control", "period_d", "snr", "surviving_periods", "error")} for k, v in out.items()})
    return ok, out


def stage_select(a, run):
    from .batch_runner import stage_select as sel

    s = sel(run, a.tics, a.sector, a.camera, a.ccd, a.cuts, not a.no_known_filter, a.max_stars, [])
    if not s.get("ok"):
        raise RuntimeError(f"select failed: {s.get('error')}")
    return s


def lc_ok(run, tic):
    p = run.path("lightcurves", f"TIC{tic}")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d if d.get("ok") and d.get("files") else None


def cov_ok(run, tic):
    p = run.path("coverage", f"TIC{tic}")
    return p.exists() and json.loads(p.read_text()).get("ok")


def run_hunt(a) -> int:
    t_start = time.time()
    deadline = t_start + a.budget if a.budget else None
    run_dir = Path(a.run_dir)
    run = Run(run_dir, cache_dir=a.cache_dir, deadline=deadline)
    prog = Progress(run_dir)
    prog.d["args"] = {k: v for k, v in vars(a).items() if k not in ("func",)}
    write_json(run_dir / "hunt_args.json", prog.d["args"])
    only = set(a.only or STAGES)
    paused = None

    def step(name):
        return name in only and paused is None

    try:
        if step("controls") and not a.no_controls:
            ok, _ = stage_controls(a, prog, deadline)
            if not ok:
                print(f"CONTROL FAILED: see {prog.path}. Fix the tools before hunting (or --no-controls).")
                return 2
        sel = stage_select(a, run)
        stars = {int(s["tic"]): s for s in sel["selected"]}
        prog.set("select", status="done", n_input=sel["n_input"], n_pass_cuts=sel["n_pass_cuts"],
                 n_known_removed=sel["n_known_removed"], n_selected=len(stars))
        tics = sorted(stars)

        if step("fetch") and not pool(run, "lightcurves", [(f"TIC{t}", dict(run_dir=run_dir, cache_dir=a.cache_dir, tic=t,
                                                                            cuts=a.cuts)) for t in tics],
                                       _fetch_job, a.dl_threads, "thread", deadline, prog, label="fetch"):
            paused = "fetch"
        lcs = {t: lc_ok(run, t) for t in tics}

        if step("dips") and not a.skip_duo:
            jobs = [(f"TIC{t}", dict(run_dir=run_dir, cache_dir=a.cache_dir, tic=t, files=lcs[t]["files"], snr_min=a.snr_min))
                    for t in tics if lcs.get(t) and cov_ok(run, t)]
            if not pool(run, "dips", jobs, _dips_job, a.workers, "process", deadline, prog):
                paused = "dips"

        if step("duo") and not a.skip_duo:
            from .batch_runner import run_batch

            prog.set("duo", status="running")
            r = run_batch(run_dir, a.tics, a.sector, a.camera, a.ccd, a.cuts, not a.no_known_filter, a.max_stars, None,
                          a.snr_min, a.duo_pmin, cache_dir=a.cache_dir, packets=not a.no_duo_packets, deadline=deadline,
                          pair_sector_rule=False)
            keep = ("n_dips", "n_dip_fragments_merged", "n_pairs", "n_duo_gate_pass", "n_demoted_star_periodic",
                    "n_duo_candidates_for_review")
            prog.set("duo", status="done", funnel={k[2:]: r["funnel"].get(k) for k in keep})

        if step("bls") and not a.skip_bls:
            jobs = [(f"TIC{t}", dict(run_dir=run_dir, cache_dir=a.cache_dir, tic=t, files=lcs[t]["files"],
                                     rstar=stars[t].get("rstar"), rho=stars[t].get("rho_solar"), pmin=a.pmin, pmax=a.pmax,
                                     n_signals=a.n_signals, grid_oversample=a.grid_oversample))
                    for t in tics if lcs.get(t)]
            if not pool(run, "bls", jobs, _bls_job, a.workers, "process", deadline, prog):
                paused = "bls"

        tri = None
        if step("triage") and not a.skip_bls:
            from .bls_triage import load_run, triage

            tri = triage(load_run(run_dir), out_dir=run_dir)
            prog.set("triage", status="done", n_survivors=tri["n_survivors"], n_near_miss=tri["n_near_miss"],
                     fail_counts=tri["fail_counts"], n_bad_windows=len(tri["bad_windows"]))

        if step("gate") and tri is not None:
            todo = (tri["survivors"] + tri["near_misses"])[:a.max_gate]
            jobs = [(gate_key(x), dict(run_dir=run_dir, cache_dir=a.cache_dir, key=gate_key(x), tic=x["tic"],
                                       files=lcs[x["tic"]]["files"], P=x["period_d"], T0=x["epoch_btjd"],
                                       dur_h=x["duration_h"], depth=x["depth_ppm"], rho=stars[x["tic"]].get("rho_solar")))
                    for x in todo if lcs.get(x["tic"])]
            if not pool(run, "gate", jobs, _gate_job, a.workers, "process", deadline, prog):
                paused = "gate"

        if step("vet") and tri is not None:
            from .candidate_packet import build

            passes = [g for g in load_stage(run_dir, "gate") if g.get("verdict") == "pass"]
            prog.set("vet", status="running", total=len(passes))
            for g in passes:
                key = gate_key(g)

                def work(g=g):
                    files = lcs[g["tic"]]["files"]
                    secs = sorted({int(s) for s in json.loads(run.path("bls", f"TIC{g['tic']}").read_text())["sectors"]})
                    sig = {"kind": "periodic", "period_d": g["period_d"], "epoch_btjd": g["epoch_btjd"],
                           "duration_h": g["duration_h"], "depth_ppm": g["depth_ppm"],
                           "rho_solar": stars[g["tic"]].get("rho_solar"), "source": "hunt"}
                    s = build(g["tic"], sig, a.cache_dir, files, tpf_sectors=secs[-a.tpf_sectors:] if a.tpf_sectors else None,
                              out_root=run_dir / "packets", offline=False)
                    return {"ok": True, "tic": g["tic"], "period_d": g["period_d"], "status": s["status"],
                            "packet_dir": s["packet_dir"],
                            **{k: s["results"][k].get("verdict") for k in ("allsector_gate", "catalog_veto", "centroid_check")},
                            "catalog_flags": [k for k, v in (s["results"]["catalog_veto"].get("flags") or {}).items() if v],
                            "secondary": s["results"]["secondary_dilution"]["flags"]}
                run.cached("vet", key, work)
            prog.set("vet", status="done", total=len(passes), done=len(passes))
    except Deferred as d:
        paused = paused or f"budget ({d})"

    if paused is None or "summary" in only:
        md = write_summary(a, run_dir, prog, paused)
        prog.set("summary", status="done", path=str(md))
    el = time.time() - t_start
    if paused:
        stg = {k: f"{v.get('done', '')}/{v.get('total', '')}" for k, v in prog.d["stages"].items() if v.get("status") == "paused"}
        print(f"PAUSED at {paused} after {el:.0f}s {stg}. Re-run the same command to continue.")
        return EXIT_PAUSED
    print(f"DONE in {el:.0f}s. {run_dir / 'summary.md'}")
    print(headline(run_dir))
    return 0


def gate_key(x):
    return f"TIC{int(x['tic'])}_P{float(x['period_d']):.5f}"


def load_stage(run_dir, stage):
    return [json.loads(p.read_text()) for p in sorted((Path(run_dir) / "stages" / stage).glob("*.json"))]


# ---------------------------------------------------------------- summary
def _errors(run_dir):
    out = {}
    for st in ("coverage", "lightcurves", "dips", "bls", "gate", "duo", "packet", "vet"):
        errs = [d.get("error") for d in load_stage(run_dir, st) if d.get("ok") is False and d.get("error")]
        if errs:
            out[st] = {"n": len(errs), "examples": sorted(set(e.split(":")[0] for e in errs))[:3]}
    return out


def _bls_explains(run_dir, tic, t1, t2, dur_h, snr_min=7.0):
    p = Path(run_dir) / "stages" / "bls" / f"TIC{int(tic)}.json"
    if not p.exists():
        return None
    for s in json.loads(p.read_text()).get("signals", []):
        if (s.get("snr") or 0) < snr_min:
            continue
        P, T0 = s["period_d"], s["epoch_btjd"]
        tol = 1.5 * max(dur_h, s["duration_h"]) / 24
        if all(abs(((t - T0 + 0.5 * P) % P) - 0.5 * P) < tol for t in (t1, t2)):
            return P
    return None


def collect(run_dir):
    run_dir = Path(run_dir)
    prog = json.loads((run_dir / "progress.json").read_text()) if (run_dir / "progress.json").exists() else {"stages": {}}
    fun = json.loads((run_dir / "funnel.json").read_text()) if (run_dir / "funnel.json").exists() else {}
    tri = json.loads((run_dir / "triage.json").read_text()) if (run_dir / "triage.json").exists() else {}
    duo = [r for r in fun.get("duo", []) if r.get("verdict") == "pass" and r.get("label_kind") == "duo"]
    for r in duo:
        r["bls_period"] = _bls_explains(run_dir, r["tic"], r["t1"], r["t2"], r["duration_h"])
    gates = load_stage(run_dir, "gate")
    vets = load_stage(run_dir, "vet")
    n_lc = sum(1 for d in load_stage(run_dir, "lightcurves") if d.get("ok") and d.get("files"))
    n_bls = sum(1 for d in load_stage(run_dir, "bls") if d.get("ok"))
    return {"prog": prog, "funnel": fun.get("funnel", {}), "duo": duo, "packets": fun.get("packets", []), "triage": tri,
            "gates": gates, "vets": vets, "n_lc": n_lc, "n_bls": n_bls, "errors": _errors(run_dir)}


def headline(run_dir) -> str:
    c = collect(run_dir)
    cand = [v for v in c["vets"] if v.get("status") == "gate_passed_needs_human_review"]
    duo_c = [p for p in c["packets"] if p.get("status") == "duo_candidate_needs_human_review"]
    duo_c = [p for p in duo_c if not any(r["tic"] == p["tic"] and r.get("bls_period") for r in c["duo"])]
    return (f"periodic candidates for review: {len(cand)} | duo candidates for review: {len(duo_c)} | "
            f"gate passes {sum(g.get('verdict') == 'pass' for g in c['gates'])}/{len(c['gates'])} | "
            f"BLS stars {c['n_bls']} | errors {sum(v['n'] for v in c['errors'].values())}")


def _fmt(x, nd=1):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def write_summary(a, run_dir, prog, paused):
    c = collect(run_dir)
    st, f, tri = c["prog"]["stages"], c["funnel"], c["triage"]
    where = f"S{a.sector} camera {a.camera} CCD {a.ccd}" if a.sector else f"{len(a.tics or [])} TICs"
    cand = [v for v in c["vets"] if v.get("status") == "gate_passed_needs_human_review"]
    L = [f"# Hunt summary: {where} ({today_et()})", ""]
    if paused:
        L += [f"**PAUSED at {paused}.** Partial results below; re-run `hunt run` with the same arguments.", ""]
    L.append(f"**{headline(run_dir)}**")
    L.append("")
    ctl = st.get("controls", {}).get("result") or {}
    if ctl:
        d, b = ctl.get("duo", {}), ctl.get("bls", {})
        L.append(f"- Controls (code {st['controls'].get('code_hash')}): TOI-4600 c duo {'pass' if d.get('ok') else 'FAIL'} "
                 f"(survivors {d.get('surviving_periods')}); pi Men c BLS {'pass' if b.get('ok') else 'FAIL'} "
                 f"(P {b.get('period_d')}, SNR {b.get('snr')})")
    s = st.get("select", {})
    L.append(f"- Selection: {s.get('n_input')} targets, {s.get('n_pass_cuts')} pass cuts, {s.get('n_known_removed')} known "
             f"TOI/CTOI removed, {s.get('n_selected')} hunted; {c['n_lc']} with usable light curves; {c['n_bls']} BLS-searched")
    if f:
        L.append(f"- Duo funnel: {f.get('n_early_and_late_coverage')} early+late coverage, {f.get('n_dips')} dips "
                 f"({f.get('n_dips_after_veto')} after common-mode veto, {f.get('n_dip_fragments_merged', 0)} fragments merged), "
                 f"{f.get('n_pairs')} pairs, {f.get('n_duo_gate_pass')} duo-gate passes, "
                 f"{f.get('n_demoted_star_periodic', 0)} demoted (star periodic)")
    if c["duo"]:
        L += ["", "| duo TIC | t1 | t2 | depth ppm | in sector | BLS periodic | surviving P (d) |", "|---|---|---|---|---|---|---|"]
        for r in c["duo"][:15]:
            sp = r.get("surviving_periods")
            sp = sp if isinstance(sp, str) else ", ".join(f"{x:.2f}" for x in (sp or [])[:4])
            L.append(f"| {r['tic']} | {r['t1']:.2f} | {r['t2']:.2f} | {r['depth_ppm']:.0f} | {r.get('in_sector')} | "
                     f"{_fmt(r.get('bls_period'), 4) or '-'} | {sp} |")
    if tri:
        L += ["", f"- BLS triage: {tri['n_signals']} top signals, {tri['n_survivors']} survivors, {tri['n_near_miss']} near misses. "
              f"Cut counts: {', '.join(f'{k} {v}' for k, v in sorted(tri['fail_counts'].items(), key=lambda kv: -kv[1]))}"]
        if tri["bad_windows"]:
            L.append("- Shared-epoch systematics (many stars dip together): " + ", ".join(
                f"BTJD {w['start_btjd']:.1f}-{w['end_btjd']:.1f} ({w['max_stars']} stars)" for w in tri["bad_windows"][:8]))
    if c["gates"]:
        L += ["", "| TIC | P (d) | depth ppm | gate | T14/T14(b=0) | why |", "|---|---|---|---|---|---|"]
        for g in sorted(c["gates"], key=lambda g: (g.get("verdict") != "pass", -(g.get("depth_ppm") or 0))):
            dd = g.get("duration") or {}
            L.append(f"| {g.get('tic')} | {_fmt(g.get('period_d'), 5)} | {_fmt(g.get('reference_depth_ppm') or g.get('depth_ppm'), 0)} | "
                     f"{g.get('verdict')} | {_fmt(dd.get('ratio_obs_over_b0'), 2)} | {(g.get('reasons') or [''])[0][:90]} |")
    if c["vets"]:
        L += ["", "| vetted TIC | P (d) | status | catalog | centroid | flags | packet |", "|---|---|---|---|---|---|---|"]
        for v in c["vets"]:
            L.append(f"| {v.get('tic')} | {_fmt(v.get('period_d'), 5)} | {v.get('status')} | {v.get('catalog_veto')} | "
                     f"{v.get('centroid_check')} | {', '.join(v.get('catalog_flags') or []) or '-'} | {v.get('packet_dir')} |")
    if c["errors"]:
        L += ["", "- Errors (recorded per star, run continued): " + "; ".join(
            f"{k} {v['n']} ({', '.join(v['examples'])})" for k, v in c["errors"].items())]
    L += ["", f"Files: `{run_dir}/triage.csv`, `{run_dir}/duo_pairs.csv`, `{run_dir}/funnel.csv`, `{run_dir}/gate/`, "
              f"`{run_dir}/packets/`", "", "A pass is not a planet. Every candidate needs a human review of its packet."]
    md = Path(run_dir) / "summary.md"
    md.write_text("\n".join(L) + "\n")
    write_log_draft(a, run_dir, c, cand)
    return md


def write_log_draft(a, run_dir, c, cand):
    """Same layout as the project's hunt logs, so the agent only has to edit the judgement calls."""
    st, f, tri = c["prog"]["stages"], c["funnel"], c["triage"]
    where = f"TESS S{a.sector} camera {a.camera} CCD {a.ccd}" if a.sector else "TIC list"
    result = (f"**Result: {len(cand)} periodic signal(s) passed every automatic check and need human review.**" if cand else
              "**Result: no likely planet on this CCD (automatic checks).**")
    ctl = st.get("controls", {}).get("result") or {}
    s = st.get("select", {})
    L = [f"# Hunt log: {today_et()}, {where}", "", result + " <!-- edit: add the judgement on the best signal -->", "",
         "## Setup", f"- shared-tools `hunt` (code {st.get('controls', {}).get('code_hash', '?')}), Python {sys.version.split()[0]}.",
         f"- Controls: TOI-4600 c duo {'passed' if ctl.get('duo', {}).get('ok') else 'FAILED'}; pi Men c BLS "
         f"{'recovered' if ctl.get('bls', {}).get('ok') else 'FAILED'} at {ctl.get('bls', {}).get('period_d')} d "
         f"(published 6.26782), SNR {ctl.get('bls', {}).get('snr')}.",
         f"- Selection: {s.get('n_input')} TESS-SPOC targets → {s.get('n_pass_cuts')} pass cuts "
         f"(Tmag {a.cuts['tmag_min']:g}–{a.cuts['tmag_max']:g}, Teff {a.cuts['teff_min']:g}–{a.cuts['teff_max']:g} K, "
         f"logg ≥ {a.cuts['logg_min']:g}) → {s.get('n_selected')} after removing TOI/CTOI.", "",
         f"## Duo search ({f.get('n_early_and_late_coverage', '?')} stars with early + late coverage)",
         f"- One pass, in-sector rule dropped, each pair labelled. {f.get('n_dip_fragments_merged', 0)} dip fragments merged; "
         f"{f.get('n_duo_gate_pass', 0)} duo-gate passes; {f.get('n_demoted_star_periodic', 0)} demoted because the star is periodic.", ""]
    if c["duo"]:
        L += ["| TIC | what it is |", "|---|---|"]
        for r in c["duo"][:15]:
            what = (f"explained by BLS periodic P = {r['bls_period']:.4f} d" if r.get("bls_period") else
                    f"duo, {r['depth_ppm']:.0f} ppm, surviving P {r.get('surviving_periods')}") + " <!-- edit -->"
            L.append(f"| {r['tic']} | {what} |")
        L.append("")
    if tri:
        L += [f"## BLS periodic search ({c['n_bls']} stars, pmin {a.pmin:g} d, pmax {a.pmax:g} d, {a.n_signals} signals per star)",
              f"- Triage (bls_triage defaults): {tri['n_survivors']} survivors, {tri['n_near_miss']} near misses. Cut counts: "
              + ", ".join(f"{k} {v}" for k, v in sorted(tri["fail_counts"].items(), key=lambda kv: -kv[1])) + ".", ""]
        if c["gates"]:
            L += ["| TIC | P (d) | depth (ppm) | outcome |", "|---|---|---|---|"]
            for g in sorted(c["gates"], key=lambda g: g.get("verdict") != "pass"):
                L.append(f"| {g['tic']} | {g['period_d']:.4f} | {_fmt(g.get('reference_depth_ppm') or g.get('depth_ppm'), 0)} | "
                         f"gate {g['verdict']}: {(g.get('reasons') or ['all tests pass'])[0][:110]} |")
            L.append("")
        if tri["bad_windows"]:
            L.append("- Shared-epoch systematics: " + ", ".join(f"BTJD {w['start_btjd']:.1f}–{w['end_btjd']:.1f}"
                                                              for w in tri["bad_windows"][:8]) + ".")
    if c["errors"]:
        L += ["", "## Errors", *[f"- {k}: {v['n']} ({', '.join(v['examples'])})" for k, v in c["errors"].items()]]
    L += ["", "## Tool issues found", "- <!-- edit -->"]
    (Path(run_dir) / "hunt-log-draft.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- status / detach / CLI
def status(run_dir) -> int:
    p = Path(run_dir) / "progress.json"
    if not p.exists():
        print("no progress.json yet")
        return 1
    d = json.loads(p.read_text())
    alive = False
    try:
        os.kill(int(d.get("pid", 0)), 0)
        alive = True
    except (OSError, ValueError):
        pass
    print(f"pid {d.get('pid')} {'running' if alive else 'not running'}; updated {d.get('updated_at')}")
    for k in STAGES:
        v = d["stages"].get(k)
        if v:
            extra = {kk: vv for kk, vv in v.items() if kk in ("n_selected", "n_survivors", "n_near_miss", "funnel")}
            dn = f" {v.get('done')}/{v.get('total')}" if "total" in v else ""
            print(f"  {k:9s} {v.get('status', '')}{dn} {json.dumps(extra) if extra else ''}".rstrip())
    if d.get("crashes"):
        print(f"  crashes: {len(d['crashes'])} (first: {d['crashes'][0]})")
    if (Path(run_dir) / "summary.md").exists():
        print(headline(run_dir))
    return 0


def detach(argv, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    args = [x for x in argv if x != "--detach"]
    if "--budget" in args:
        i = args.index("--budget")
        del args[i:i + 2]
    log = open(run_dir / "hunt.log", "ab")
    p = subprocess.Popen([sys.executable, "-m", "shared_tools.exo.hunt", *args, "--budget", "0"], stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, start_new_session=True, cwd=os.getcwd())
    print(f"started pid {p.pid}; log {run_dir / 'hunt.log'}; check with: hunt status --run-dir {run_dir}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--sector", type=int)
    r.add_argument("--camera", type=int)
    r.add_argument("--ccd", type=int)
    r.add_argument("--tics", type=lambda s: [int("".join(ch for ch in x if ch.isdigit())) for x in s.split(",") if x])
    r.add_argument("--max-stars", type=int)
    r.add_argument("--budget", type=float, default=150.0, help="seconds before pausing (0 = no limit). Default 150")
    r.add_argument("--detach", action="store_true", help="run to completion in the background")
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)), help="CPU processes")
    r.add_argument("--dl-threads", type=int, default=4, help="parallel downloads (MAST throttles at ~12)")
    r.add_argument("--only", action="append", choices=STAGES, help="run only these stages (repeatable)")
    r.add_argument("--no-controls", action="store_true")
    r.add_argument("--skip-duo", action="store_true")
    r.add_argument("--skip-bls", action="store_true")
    r.add_argument("--no-duo-packets", action="store_true")
    r.add_argument("--no-known-filter", action="store_true")
    r.add_argument("--snr-min", type=float, default=7.0, help="single-dip SNR for the duo funnel")
    r.add_argument("--duo-pmin", type=float, default=10.0)
    r.add_argument("--pmin", type=float, default=BLS["pmin"])
    r.add_argument("--pmax", type=float, default=BLS["pmax"])
    r.add_argument("--n-signals", type=int, default=BLS["n_signals"])
    r.add_argument("--grid-oversample", type=float)
    r.add_argument("--max-gate", type=int, default=40)
    r.add_argument("--tpf-sectors", type=int, default=4, help="centroid on the most recent N sectors (0 = none)")
    for k, v in CUTS.items():
        r.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    r.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    s = sub.add_parser("status")
    s.add_argument("--run-dir", type=Path, required=True)
    m = sub.add_parser("summary")
    m.add_argument("--run-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "status":
        return status(a.run_dir)
    if a.cmd == "summary":
        print((Path(a.run_dir) / "summary.md").read_text())
        return 0
    if not (a.tics or (a.sector and a.camera and a.ccd)):
        ap.error("give --tics, or --sector with --camera and --ccd")
    if a.detach:
        return detach(argv, a.run_dir)
    a.cuts = {k: getattr(a, k) for k in CUTS}
    return run_hunt(a)


if __name__ == "__main__":
    raise SystemExit(main())
