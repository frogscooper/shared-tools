"""candidate_packet: one folder a vetter can review without rerunning anything.

Inputs: TIC + a signal (a transit_search signals.json with --rank, or --period/--epoch/--duration),
light curves (tess_lc_fetch, or --lc-file) and optionally target pixel files (--tpf-sectors
fetches them; --tpf-file uses local ones).

Writes <out>/TIC<id>_P<period>/:
  packet.md, signal.json, full_lc.png, fold.png, odd_even.png, events.png,
  gate.png (allsector_gate), secondary.png (secondary_dilution), centroid_s<N>.png (centroid_check)
and adds or replaces this signal's row in <out>/candidates.csv.

Single dips and duos (two single transits) get their own packet, written to TIC<id>_single_<epoch>/
or TIC<id>_duo_<t1>_<t2>/: packet.md, signal.json, full_lc.png (events marked), events.png (each event
zoomed with its trapezoid and V fits), aliases.png + alias table and predicted windows (duo, from
duo_solver), and centroid_e<i>_s<N>.png (difference image per event, from centroid_check).

Status is never "planet". At best it is "gate_passed_needs_human_review" (periodic),
"duo_candidate_needs_human_review" or "single_needs_human_review".

    candidate-packet 261136679 --signals ~/.cache/shared_tools/transit_search/TIC261136679/signals.json --rank 1 --tpf-sectors 1,4,8
    candidate-packet 232608943 --t1 1785.96 --t2 2751.5986 --duration 10.85 --depth 12000 --tpf-sectors 17,53
    candidate-packet 230064179 --single-epoch 3515.8593 --duration 5.38 --depth 1996 --offline
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from .allsector_gate import gate, load_sectors, prepare_sectors
from .common import default_cache_dir, jsonable, now_et, parse_tic, write_json

CSV_COLS = ["tic", "kind", "period_d", "epoch_btjd", "duration_h", "depth_ppm", "snr", "gate_verdict",
            "centroid_verdict", "secondary_flag", "neighbor_could_cause", "catalog_verdict", "status",
            "packet_dir", "generated_at", "t2_btjd", "surviving_periods"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def bin_xy(x, y, nb=80, lim=None):
    lim = lim or (x.min(), x.max())
    edges = np.linspace(lim[0], lim[1], nb + 1)
    idx = np.digitize(x, edges)
    xb = np.array([x[idx == i].mean() for i in range(1, nb + 1) if (idx == i).sum() > 2])
    yb = np.array([y[idx == i].mean() for i in range(1, nb + 1) if (idx == i).sum() > 2])
    return xb, yb


def plot_full(secs, P, T0, path):
    plt = _plt()
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    for s in secs:
        g = s["good"]
        ax[0].plot(s["time"][g], s["flux"][g] / np.nanmedian(s["flux"][g]), ",", color="k", alpha=0.4)
        ax[1].plot(s["time"][g], s["flat"][g], ",", color="k", alpha=0.4)
        ax[0].text(np.nanmedian(s["time"]), ax[0].get_ylim()[1] if False else 1.0, f"S{s['sector']}", fontsize=6)
    t = np.concatenate([s["time"] for s in secs])
    for tc in T0 + P * np.arange(np.floor((np.nanmin(t) - T0) / P), np.ceil((np.nanmax(t) - T0) / P) + 1):
        if any(np.nanmin(s["time"]) <= tc <= np.nanmax(s["time"]) for s in secs):
            ax[1].axvline(tc, color="r", lw=0.3, alpha=0.5)
    ax[0].set_ylabel("normalized flux")
    ax[1].set(ylabel="detrended", xlabel="BTJD", title="red: predicted transits")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def fold_arrays(secs, P, T0):
    t = np.concatenate([s["time"][s["good"]] for s in secs])
    f = np.concatenate([s["flat"][s["good"]] for s in secs])
    ok = np.isfinite(f)
    t, f = t[ok], f[ok]
    x = (((t - T0) / P + 0.5) % 1 - 0.5) * P * 24
    n = np.round((t - T0) / P).astype(int)
    return x, f, n


def plot_fold(secs, P, T0, dur_h, path):
    plt = _plt()
    x, f, _ = fold_arrays(secs, P, T0)
    w = np.abs(x) < 3 * dur_h
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x[w], (f[w] - 1) * 1e6, ",", color="gray", alpha=0.5)
    xb, yb = bin_xy(x[w], (f[w] - 1) * 1e6, 60, (-3 * dur_h, 3 * dur_h))
    ax.plot(xb, yb, "o", color="C0", ms=4)
    lo, hi = np.percentile((f[w] - 1) * 1e6, [1, 99])
    ax.set(ylim=(lo, hi), xlabel="hours from mid-transit", ylabel="flux - 1 [ppm]", title=f"fold P={P:.5f} d")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_odd_even(secs, P, T0, dur_h, path):
    plt = _plt()
    x, f, n = fold_arrays(secs, P, T0)
    w = np.abs(x) < 3 * dur_h
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, par, name in ((ax[0], 1, "odd"), (ax[1], 0, "even")):
        m = w & (n % 2 == par)
        if m.sum() > 10:
            xb, yb = bin_xy(x[m], (f[m] - 1) * 1e6, 50, (-3 * dur_h, 3 * dur_h))
            a.plot(xb, yb, "o", ms=4)
            core = m & (np.abs(x) < 0.4 * dur_h)
            if core.any():
                a.axhline(-(1 - np.mean(f[core])) * 1e6, color="r", ls="--", lw=0.8)
        a.set(title=name, xlabel="hours from mid-transit")
    ax[0].set_ylabel("flux - 1 [ppm]")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_events(secs, events, dur_h, path, max_events=12):
    plt = _plt()
    use = [e for e in events if e["status"] != "no_coverage"][:max_events]
    if not use:
        return None
    nc = 4
    nr = int(np.ceil(len(use) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(3 * nc, 2.4 * nr), squeeze=False)
    by_sec = {s["sector"]: s for s in secs}
    for a, e in zip(ax.flat, use):
        s = by_sec[e["sector"]]
        dt = (s["time"] - e["t_pred_btjd"]) * 24
        m = s["good"] & (np.abs(dt) < 3 * dur_h)
        a.plot(dt[m], (s["flat"][m] - 1) * 1e6, ".", ms=1.5, color="k")
        a.axvspan(-dur_h / 2, dur_h / 2, color="r", alpha=0.1)
        a.set_title(f"S{e['sector']} E{e['epoch']} {e['status']}\n{e['depth_ppm']}±{e['depth_err_ppm']} ppm", fontsize=7)
        a.tick_params(labelsize=6)
    for a in list(ax.flat)[len(use):]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def load_signal(signals_json=None, rank=1, period=None, epoch=None, duration=None, depth=None, t1=None, t2=None,
                single_epoch=None):
    if signals_json:
        d = json.loads(Path(signals_json).read_text())
        sigs = d["signals"] if "signals" in d else [d]
        sig = next(s for s in sigs if s.get("rank", 1) == rank)
    elif t1 is not None and t2 is not None:
        sig = {"kind": "duo", "t1": t1, "t2": t2, "duration_h": duration, "depth_ppm": depth, "source": "command line"}
    elif single_epoch is not None:
        sig = {"kind": "single", "epoch_btjd": single_epoch, "duration_h": duration, "depth_ppm": depth,
               "source": "command line"}
    else:
        sig = {"kind": "periodic", "period_d": period, "epoch_btjd": epoch, "duration_h": duration, "depth_ppm": depth,
               "status": "unpromoted", "source": "command line"}
    if sig.get("kind") not in ("single", "duo") and not sig.get("period_d"):
        raise ValueError("a periodic signal needs period_d; use kind single (--single-epoch) or duo (--t1/--t2)")
    if sig.get("duration_h") is None:
        raise ValueError("signal needs duration_h")
    return sig


def update_candidates_csv(path: Path, row: dict):
    """Insert or replace (same tic, kind, period to 1e-4 d). Idempotent."""
    rows = []
    if path.exists():
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
    def k(r):
        x = r.get("period_d") if r["kind"] not in ("single", "duo") else r.get("epoch_btjd")
        return (str(r["tic"]), r["kind"], f"{float(x):.4f}" if x not in (None, "") else "")
    key = k(row)
    rows = [r for r in rows if k(r) != key] + [jsonable(row)]
    tmp = path.with_suffix(".part")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def build(tic, signal: dict, cache_dir=None, lc_files=None, tpf_sectors=None, tpf_files=None, out_root=None,
          claimed_sectors=None, offline=False, neighbors=None, secs=None, duo=None) -> dict:
    if signal.get("kind") in ("single", "duo"):
        return build_events(tic, signal, cache_dir, lc_files, tpf_sectors, tpf_files, out_root, offline, neighbors, secs, duo)
    tic = parse_tic(tic)
    cache_dir = Path(cache_dir or default_cache_dir())
    P, T0, dur_h = float(signal["period_d"]), float(signal["epoch_btjd"]), float(signal["duration_h"])
    out_root = Path(out_root or cache_dir / "packets")
    d = out_root / f"TIC{tic}_P{P:.5f}"
    d.mkdir(parents=True, exist_ok=True)
    notes, sub = [], {}

    # 1. gate (also gives per-event depths for the zooms)
    g = gate(tic, P, T0, dur_h, signal.get("depth_ppm"), claimed_sectors, cache_dir, lc_files, d / "_gate",
             rho_solar=signal.get("rho_solar"))
    shutil.copy(g["plot"], d / "gate.png")
    sub["allsector_gate"] = {"verdict": g["verdict"], "reasons": g["reasons"],
                             "tests": {k: v["result"] for k, v in g["tests"].items()}, "json": g["json"]}

    # 2. light-curve figures
    secs, _ = load_sectors(tic, cache_dir, lc_files)
    prepare_sectors(secs, P, T0, dur_h / 24)
    plot_full(secs, P, T0, d / "full_lc.png")
    plot_fold(secs, P, T0, dur_h, d / "fold.png")
    plot_odd_even(secs, P, T0, dur_h, d / "odd_even.png")
    plot_events(secs, g["events"], dur_h, d / "events.png")

    # 3. catalog + neighbors (network; fail-closed result recorded as-is)
    if offline or neighbors is not None:
        sub["catalog_veto"] = {"verdict": "not_run", "reason": "offline" if offline else "neighbors supplied by caller"}
        neighbors = neighbors or []
        notes.append("catalog_veto not run; run it before promoting this signal")
    else:
        from .catalog_veto import veto

        v = veto(tic, period_d=P, epoch_btjd=T0, depth_ppm=g["reference_depth_ppm"], cache_dir=cache_dir)
        sub["catalog_veto"] = {"verdict": v["summary"]["verdict"], "reason": v["summary"]["reason"],
                               "flags": v["flags"], "sources_down": v["summary"]["sources_down"]}
        neighbors = v["sources"].get("gaia_dr3_neighbors", {}).get("matches", [])

    # 4. secondary + dilution
    from .secondary_dilution import run as sec_run

    sd = sec_run(tic, P, T0, dur_h, g["reference_depth_ppm"], neighbors, cache_dir, lc_files, d / "_secondary")
    shutil.copy(sd["plot"], d / "secondary.png")
    sub["secondary_dilution"] = {"flags": sd["flags"], "phase_0p5": sd["secondary"]["phase_0p5"],
                                 "strongest_other_phase": sd["secondary"]["strongest_other_phase"],
                                 "n_neighbors_could_cause": sd["dilution"]["n_could_cause"],
                                 "could_cause": sd["dilution"]["could_cause"][:5], "json": sd["json"]}

    # 5. centroid if TPFs are available
    if tpf_files or (tpf_sectors and not offline):
        from .centroid_check import check

        c = check(tic, P, T0, dur_h, tpf_sectors, cache_dir=cache_dir, tpf_files=tpf_files, out_dir=d / "_centroid")
        for r in c["tpfs"]:
            if r.get("plot"):
                shutil.copy(r["plot"], d / Path(r["plot"]).name)
        sub["centroid_check"] = {"verdict": c["verdict"], "basis": c["verdict_basis"],
                                 "combined_photocenter": c["combined_photocenter"],
                                 "per_tpf": [{k: r.get(k) for k in ("sector", "verdict", "method", "centroid_offset_px",
                                                                    "peak_offset_px", "diff_snr_at_target")} for r in c["tpfs"]],
                                 "json": c["json"]}
    else:
        sub["centroid_check"] = {"verdict": "not_run", "reason": "no TPF requested"}
        notes.append("no centroid check: pass --tpf-sectors or --tpf-file")

    cat = sub["catalog_veto"]["verdict"]
    cen = sub["centroid_check"]["verdict"]
    if cat == "known_object":
        status = "known_object"
    elif g["verdict"] == "pass" and cen == "on_target" and not sd["flags"]["secondary_at_0p5"]:
        status = "gate_passed_needs_human_review"
    else:
        status = "unpromoted"
    summary = {"tool": "candidate_packet", "tic": tic, "generated_at": now_et(), "status": status,
               "signal": signal, "results": sub, "notes": notes,
               "disclaimer": "Review packet only. Not a detection or discovery claim."}
    write_json(d / "signal.json", summary)
    write_packet_md(d, summary, g)
    row = {"tic": tic, "kind": signal.get("kind", "periodic"), "period_d": P, "epoch_btjd": T0, "duration_h": dur_h,
           "depth_ppm": g["reference_depth_ppm"], "snr": signal.get("snr"), "gate_verdict": g["verdict"],
           "centroid_verdict": cen, "secondary_flag": sd["flags"]["secondary_at_0p5"],
           "neighbor_could_cause": sd["flags"]["neighbor_could_cause"], "catalog_verdict": cat, "status": status,
           "packet_dir": str(d), "generated_at": summary["generated_at"]}
    update_candidates_csv(out_root / "candidates.csv", row)
    summary["packet_dir"] = str(d)
    summary["files"] = sorted(p.name for p in d.iterdir() if p.is_file())
    return summary


# ---------------------------------------------------------------- single dips and duos
def plot_event_zooms(secs, events, dur_h, path):
    """One panel per event: detrended flux, binned points, trapezoid (red) and V (blue dashed) fits."""
    from .duo_solver import binned_model, event_series

    plt = _plt()
    fig, ax = plt.subplots(1, len(events), figsize=(5 * len(events), 3.6), squeeze=False)
    for a, e in zip(ax.flat, events):
        tc = e.get("t0_btjd") or e["t_input_btjd"]
        t, f, s = event_series(secs, tc, dur_h / 24)
        if t is None:
            a.set_title(f"event at {tc}: no data")
            continue
        m = np.abs(t - tc) < 2.5 * dur_h / 24
        x = (t[m] - tc) * 24
        a.plot(x, (f[m] - 1) * 1e6, ".", ms=2, color="0.5")
        if m.sum() > 20:
            xb, yb = bin_xy(x, (f[m] - 1) * 1e6, 40)
            a.plot(xb, yb, "o", ms=3, color="k")
        if e.get("status") == "ok":
            tt = np.linspace(t[m].min(), t[m].max(), 600)
            cad = float(np.median(np.diff(t))) if len(t) > 2 else 0
            for tau, sty, lab in ((e["ingress_frac"] * e["t14_h"] / 24, "r-", "trapezoid"), (e["t14_h"] / 48, "b--", "V")):
                mod = binned_model(tt, e["t0_btjd"], e["depth_ppm"] * 1e-6, e["t14_h"] / 24, tau, cad)
                a.plot((tt - tc) * 24, (mod - 1) * 1e6, sty, lw=1, label=lab)
            a.legend(fontsize=6)
        a.set_title(f"S{s['sector']} BTJD {tc:.4f}\n{e.get('depth_ppm')}±{e.get('depth_err_ppm')} ppm, "
                    f"T14 {e.get('t14_h')} h, {e.get('shape')}", fontsize=8)
        a.set_xlabel("hours from event")
    ax.flat[0].set_ylabel("flux - 1 [ppm]")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_events_full(secs, times, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, 3.5))
    for s in secs:
        g = s["good"] & np.isfinite(s["flat"])
        ax.plot(s["time"][g], s["flat"][g], ",", color="k", alpha=0.4)
        ax.text(np.nanmedian(s["time"]), np.nanmax(s["flat"][g]) if g.any() else 1, f"S{s['sector']}", fontsize=6)
    for tc in times:
        ax.axvline(tc, color="r", lw=0.8, alpha=0.6)
    ax.set(xlabel="BTJD", ylabel="detrended", title="red: the event(s) in this packet")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def plot_aliases(duo, path):
    plt = _plt()
    colors = {"excluded": "0.7", "open": "C0", "seen": "C2", "unresolved": "C1"}
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for st, c in colors.items():
        a = [x for x in duo["aliases"] if x["status"] == st]
        if a:
            ax.scatter([x["period_d"] for x in a], [x["n_windows_in_coverage"] for x in a], c=c, s=18, label=f"{st} ({len(a)})")
    ax.set(xscale="log", xlabel="alias period (d)", ylabel="predicted transits in coverage",
           title=f"duo aliases P = {duo['delta_t_d']} / n")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def build_events(tic, signal, cache_dir=None, lc_files=None, tpf_sectors=None, tpf_files=None, out_root=None, offline=False,
                 neighbors=None, secs=None, duo=None) -> dict:
    """Packet for one single dip (kind single) or two (kind duo)."""
    from .duo_solver import event_series, fit_event, prepare, solve

    tic = parse_tic(tic)
    cache_dir = Path(cache_dir or default_cache_dir())
    kind = signal["kind"]
    dur_h = float(signal["duration_h"])
    depth = float(signal.get("depth_ppm") or 1000.0)
    times = [float(signal["epoch_btjd"])] if kind == "single" else sorted([float(signal["t1"]), float(signal["t2"])])
    out_root = Path(out_root or cache_dir / "packets")
    d = out_root / (f"TIC{tic}_single_{times[0]:.4f}" if kind == "single" else f"TIC{tic}_duo_{times[0]:.3f}_{times[1]:.3f}")
    d.mkdir(parents=True, exist_ok=True)
    notes, sub = [], {}
    if secs is None:
        secs, _ = load_sectors(tic, cache_dir, lc_files)

    # 1. duo: alias solver + duo gate (also refits both events); single: event fit only
    if kind == "duo":
        duo = duo or solve(tic, times[0], times[1], dur_h, depth, signal.get("t1_err", 0.01), signal.get("t2_err", 0.01),
                           signal.get("pmin", 10.0), signal.get("rho_solar"), cache_dir, secs=secs, out_dir=d / "_duo")
        events = duo["events"]
        times = [e.get("t0_btjd") or e["t_input_btjd"] for e in events]
        dur_h = duo["inputs"]["duration_h"]
        plot_aliases(duo, d / "aliases.png")
        sub["duo_solver"] = {"verdict": duo["gate"]["verdict"], "tests": duo["gate"]["tests"], "label": duo["label"],
                             "n_aliases": duo["n_aliases"], "n_excluded": duo["n_excluded"],
                             "surviving": [{k: a.get(k) for k in ("n", "period_d", "period_err_d", "status",
                                                                  "n_windows_in_coverage", "n_seen", "expected_t14_h_b0")}
                                           | {"next_windows": a.get("next_windows", [])[:2]}
                                           for a in duo["aliases"] if a["status"] != "excluded"],
                             "json": duo.get("json")}
    else:
        prepare(secs, times, dur_h / 24)
        t, f, s = event_series(secs, times[0], dur_h / 24)
        e = fit_event(t, f, times[0], dur_h / 24, depth * 1e-6) if t is not None else {"status": "no_data"}
        e.update(t_input_btjd=times[0], sector=s["sector"] if s else None)
        events = [e]
        if e.get("status") == "ok":
            times, dur_h = [e["t0_btjd"]], e["t14_h"]
        sub["event_shape"] = {"verdict": "pass" if e.get("planet_shaped") else "fail", "shape": e.get("shape")}
    prepare(secs, times, dur_h / 24)
    plot_events_full(secs, times, d / "full_lc.png")
    plot_event_zooms(secs, events, dur_h, d / "events.png")

    # 2. catalog + neighbors
    ref_depth = float(np.mean([e["depth_ppm"] for e in events if e.get("status") == "ok"] or [depth]))
    if offline or neighbors is not None:
        sub["catalog_veto"] = {"verdict": "not_run", "reason": "offline" if offline else "neighbors supplied by caller"}
        neighbors = neighbors or []
        notes.append("catalog_veto not run; run it before promoting this signal")
    else:
        from .catalog_veto import veto

        v = veto(tic, depth_ppm=ref_depth, cache_dir=cache_dir)
        sub["catalog_veto"] = {"verdict": v["summary"]["verdict"], "reason": v["summary"]["reason"],
                               "flags": v["flags"], "sources_down": v["summary"]["sources_down"]}
        neighbors = v["sources"].get("gaia_dr3_neighbors", {}).get("matches", [])
    from .secondary_dilution import dilution_table

    dil = dilution_table(neighbors, ref_depth)
    could = [r for r in dil if r["could_cause"]]
    sub["dilution"] = {"n_neighbors": len(dil), "n_could_cause": len(could), "could_cause": could[:5]}

    # 3. per-event centroid (one transit per TPF: period set far beyond the baseline)
    per = []
    if tpf_files or (tpf_sectors and not offline):
        from .centroid_check import check

        for i, e in enumerate(events, 1):
            sec = e.get("sector")
            if tpf_sectors and sec not in tpf_sectors and not tpf_files:
                continue
            try:
                c = check(tic, 1e5, times[i - 1], dur_h, [sec] if sec else None, cache_dir=cache_dir, tpf_files=tpf_files,
                          out_dir=d / f"_centroid_e{i}", oot_local_d=max(1.0, 4 * dur_h / 24))
            except Exception as exc:  # noqa: BLE001 - record and continue, verdict stays not on_target
                per.append({"event": i, "sector": sec, "verdict": "error", "error": f"{type(exc).__name__}: {exc}"[:200]})
                continue
            for r in c["tpfs"]:
                if r.get("plot"):
                    shutil.copy(r["plot"], d / f"centroid_e{i}_s{r['sector']}.png")
                per.append({"event": i, **{k: r.get(k) for k in ("sector", "verdict", "method", "centroid_offset_px",
                                                                 "peak_offset_px", "diff_snr_at_target")},
                            "photocenter": r.get("photocenter")})
    verdicts = [p["verdict"] for p in per]
    cen = ("not_run" if not per else "off_target" if "off_target" in verdicts else
           "on_target" if len(per) >= len(events) and all(v == "on_target" for v in verdicts) else "inconclusive")
    sub["centroid_check"] = {"verdict": cen, "per_event": per}
    if cen == "not_run":
        notes.append("no centroid check: pass --tpf-sectors (the event sectors) or --tpf-file")

    cat = sub["catalog_veto"]["verdict"]
    if cat == "known_object":
        status = "known_object"
    elif kind == "duo":
        ok = duo["gate"]["verdict"] == "pass" and duo["gate"]["tests"]["label"]["result"] == "duo"
        status = "duo_candidate_needs_human_review" if ok and cen == "on_target" and cat == "clean" else "unpromoted"
    else:
        status = ("single_needs_human_review" if sub["event_shape"]["verdict"] == "pass" and cen == "on_target"
                  and cat == "clean" else "unpromoted")
    if signal.get("common_mode"):
        status = "unpromoted"
        notes.append("cross_star_veto flagged this epoch as common-mode on the CCD")
    summary = {"tool": "candidate_packet", "kind": kind, "tic": tic, "generated_at": now_et(), "status": status,
               "signal": signal, "events": events, "results": sub, "notes": notes,
               "disclaimer": "Review packet only. Not a detection or discovery claim."}
    write_json(d / "signal.json", summary)
    write_events_md(d, summary)
    surv = sub.get("duo_solver", {}).get("surviving", [])
    row = {"tic": tic, "kind": kind, "period_d": "", "epoch_btjd": times[0], "t2_btjd": times[1] if kind == "duo" else "",
           "duration_h": dur_h, "depth_ppm": round(ref_depth, 1), "snr": signal.get("snr"),
           "gate_verdict": sub.get("duo_solver", sub.get("event_shape", {})).get("verdict"), "centroid_verdict": cen,
           "secondary_flag": "", "neighbor_could_cause": bool(could), "catalog_verdict": cat, "status": status,
           "surviving_periods": ";".join(str(a["period_d"]) for a in surv), "packet_dir": str(d),
           "generated_at": summary["generated_at"]}
    update_candidates_csv(out_root / "candidates.csv", row)
    summary["packet_dir"] = str(d)
    summary["files"] = sorted(p.name for p in d.iterdir() if p.is_file())
    return summary


def write_events_md(d: Path, s: dict):
    r = s["results"]
    title = "single dip" if s["kind"] == "single" else "duo (two single transits)"
    lines = [f"# TIC {s['tic']}: {title} review packet", "", f"**Status: {s['status']}.** {s['disclaimer']}", "",
             "## Events", "", "| # | sector | t0 (BTJD) | t0 err (d) | depth (ppm) | T14 (h) | shape | partial |",
             "|---|---|---|---|---|---|---|---|"]
    for i, e in enumerate(s["events"], 1):
        lines.append(f"| {i} | {e.get('sector')} | {e.get('t0_btjd', e.get('t_input_btjd'))} | {e.get('t0_err_d')} | "
                     f"{e.get('depth_ppm')} ± {e.get('depth_err_ppm')} | {e.get('t14_h')} ± {e.get('t14_err_h')} | "
                     f"{e.get('shape')} (ingress/T14 {e.get('ingress_frac')}, dchi2 V-trap {e.get('dchi2_v_minus_trap')}) | "
                     f"{e.get('partial')} |")
    lines += ["", "![full](full_lc.png)", "", "![events](events.png)", ""]
    if "duo_solver" in r:
        du = r["duo_solver"]
        lines += [f"## Duo gate: **{du['verdict']}**", "", f"Label: **{du['label']}**", "",
                  "| test | result | why |", "|---|---|---|"]
        lines += [f"| {k} | {v['result']} | {'; '.join(v.get('why', [])) if isinstance(v.get('why'), list) else ''} |"
                  for k, v in du["tests"].items()]
        lines += ["", f"{du['n_aliases']} aliases, {du['n_excluded']} excluded by measured data. Surviving:", "",
                  "| n | period (d) | status | windows in coverage | expected T14 b=0 (h) | next window (UTC) |",
                  "|---|---|---|---|---|---|"]
        for a in du["surviving"]:
            nw = (a["next_windows"] or [{}])[0]
            lines.append(f"| {a['n']} | {a['period_d']} ± {a['period_err_d']} | {a['status']} | {a['n_windows_in_coverage']} | "
                         f"{a.get('expected_t14_h_b0')} | {nw.get('window_start_utc')} to {nw.get('window_end_utc')} |")
        lines += ["", "![aliases](aliases.png)", ""]
    else:
        lines += [f"## Event shape: **{r['event_shape']['verdict']}** ({r['event_shape']['shape']})", ""]
    c = r["centroid_check"]
    lines += [f"## Centroid per event: **{c['verdict']}**", ""]
    if c["per_event"]:
        lines += ["| event | sector | verdict | method | 4-px centroid offset (px) | brightest diff px (px) | diff SNR |",
                  "|---|---|---|---|---|---|---|"]
        lines += [f"| {p['event']} | {p.get('sector')} | {p['verdict']} | {p.get('method')} | {p.get('centroid_offset_px')} | "
                  f"{p.get('peak_offset_px')} | {p.get('diff_snr_at_target')} |" for p in c["per_event"]]
        lines += [""] + [f"![centroid](centroid_e{p['event']}_s{p['sector']}.png)" for p in c["per_event"]
                         if (d / f"centroid_e{p['event']}_s{p['sector']}.png").exists()]
    lines += ["", f"## Dilution: {r['dilution']['n_could_cause']} of {r['dilution']['n_neighbors']} Gaia neighbors could "
              f"produce this depth", "", f"## Catalogs: **{r['catalog_veto']['verdict']}**", "", f"{r['catalog_veto']}", ""]
    if s["notes"]:
        lines += ["## Notes", ""] + [f"- {n}" for n in s["notes"]]
    (d / "packet.md").write_text("\n".join(lines) + "\n")


def write_packet_md(d: Path, s: dict, g: dict):
    r = s["results"]
    sig = s["signal"]
    lines = [
        f"# TIC {s['tic']}: review packet", "",
        f"**Status: {s['status']}.** {s['disclaimer']}", "",
        "## Signal", "",
        "| period (d) | epoch (BTJD) | duration (h) | depth (ppm) | SNR | SDE | source |", "|---|---|---|---|---|---|---|",
        f"| {sig['period_d']} | {sig['epoch_btjd']} | {sig['duration_h']} | {g['reference_depth_ppm']} | "
        f"{sig.get('snr')} | {sig.get('sde')} | {sig.get('source', 'transit_search rank ' + str(sig.get('rank')))} |", "",
        f"## All-sector gate: **{g['verdict']}**", "",
        "| test | result |", "|---|---|"] + [f"| {k} | {v} |" for k, v in r["allsector_gate"]["tests"].items()] + [
        "", "Reasons: " + ("; ".join(map(str, g["reasons"])) or "none"), "",
        f"Sectors: {', '.join(str(x['sector']) for x in g['sectors'])}. Clean events: {g['n_clean']} of {g['n_events']}.",
        "", "![gate](gate.png)", "", "## Light curve", "", "![full](full_lc.png)", "", "![fold](fold.png)", "",
        "![odd/even](odd_even.png)", "", "![events](events.png)", "",
        f"## Centroid: **{r['centroid_check']['verdict']}**", ""]
    c = r["centroid_check"]
    if c["verdict"] != "not_run":
        lines += [f"Basis: {c['basis']}. Combined photocenter: {c['combined_photocenter']}", "",
                  "| sector | verdict | method | 4-px centroid offset (px) | brightest diff px offset (px) | diff SNR at target |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {x['sector']} | {x['verdict']} | {x['method']} | {x['centroid_offset_px']} | {x['peak_offset_px']} | "
                  f"{x['diff_snr_at_target']} |" for x in c["per_tpf"]]
        lines += [""] + [f"![centroid](centroid_s{x['sector']}.png)" for x in c["per_tpf"] if (d / f"centroid_s{x['sector']}.png").exists()]
    else:
        lines.append(c["reason"])
    sd = r["secondary_dilution"]
    lines += ["", "## Secondary eclipse and dilution", "",
              f"Phase 0.5: {sd['phase_0p5']['depth_ppm']} ± {sd['phase_0p5']['err_ppm']} ppm (SNR {sd['phase_0p5']['snr']}). "
              f"Strongest other phase: {sd['strongest_other_phase']}.",
              f"Flags: {sd['flags']}. Neighbors that could cause the depth: {sd['n_neighbors_could_cause']}.", "",
              "![secondary](secondary.png)", "", f"## Catalogs: **{r['catalog_veto']['verdict']}**", "",
              f"{r['catalog_veto']}", ""]
    if s["notes"]:
        lines += ["## Notes", ""] + [f"- {n}" for n in s["notes"]]
    (d / "packet.md").write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tic")
    ap.add_argument("--signals", type=Path, help="transit_search signals.json (or a one-signal JSON)")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--period", type=float)
    ap.add_argument("--epoch", type=float)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--depth", type=float)
    ap.add_argument("--single-epoch", type=float, help="single-dip packet at this BTJD")
    ap.add_argument("--t1", type=float, help="duo packet: first event BTJD")
    ap.add_argument("--t2", type=float, help="duo packet: second event BTJD")
    ap.add_argument("--claimed-sectors", type=lambda s: [int(x) for x in s.split(",") if x])
    ap.add_argument("--tpf-sectors", type=lambda s: [int(x) for x in s.split(",") if x])
    ap.add_argument("--tpf-file", action="append")
    ap.add_argument("--lc-file", action="append")
    ap.add_argument("--offline", action="store_true", help="skip catalog queries and TPF downloads")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--out", type=Path, help="packet root (default <cache>/packets)")
    a = ap.parse_args(argv)
    sig = load_signal(a.signals, a.rank, a.period, a.epoch, a.duration, a.depth, a.t1, a.t2, a.single_epoch)
    s = build(a.tic, sig, a.cache_dir, a.lc_file, a.tpf_sectors, a.tpf_file, a.out, a.claimed_sectors, a.offline)
    r = s["results"]
    gate_v = (r.get("allsector_gate") or r.get("duo_solver") or r.get("event_shape"))["verdict"]
    print(json.dumps(jsonable({"tic": s["tic"], "status": s["status"], "packet_dir": s["packet_dir"], "files": s["files"],
                               "gate": gate_v, "centroid": r["centroid_check"]["verdict"],
                               "catalog": r["catalog_veto"]["verdict"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
