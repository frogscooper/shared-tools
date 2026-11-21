"""candidate_packet: one folder a vetter can review without rerunning anything.

Inputs: TIC + a signal (a transit_search signals.json with --rank, or --period/--epoch/--duration),
light curves (tess_lc_fetch, or --lc-file) and optionally target pixel files (--tpf-sectors
fetches them; --tpf-file uses local ones).

Writes <out>/TIC<id>_P<period>/:
  packet.md, signal.json, full_lc.png, fold.png, odd_even.png, events.png,
  gate.png (allsector_gate), secondary.png (secondary_dilution), centroid_s<N>.png (centroid_check)
and adds or replaces this signal's row in <out>/candidates.csv.

Status is never "planet". At best it is "gate_passed_needs_human_review".

    candidate-packet 261136679 --signals ~/.cache/shared_tools/transit_search/TIC261136679/signals.json --rank 1 --tpf-sectors 1,4,8
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
            "packet_dir", "generated_at"]


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


def load_signal(signals_json=None, rank=1, period=None, epoch=None, duration=None, depth=None):
    if signals_json:
        d = json.loads(Path(signals_json).read_text())
        sigs = d["signals"] if "signals" in d else [d]
        sig = next(s for s in sigs if s.get("rank", 1) == rank)
    else:
        sig = {"kind": "periodic", "period_d": period, "epoch_btjd": epoch, "duration_h": duration, "depth_ppm": depth,
               "status": "unpromoted", "source": "command line"}
    if sig.get("kind") == "single" or not sig.get("period_d"):
        raise ValueError("candidate_packet needs a periodic signal (single-transit packets are not supported yet)")
    return sig


def update_candidates_csv(path: Path, row: dict):
    """Insert or replace (same tic, kind, period to 1e-4 d). Idempotent."""
    rows = []
    if path.exists():
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
    key = (str(row["tic"]), row["kind"], f"{float(row['period_d']):.4f}")
    rows = [r for r in rows if (r["tic"], r["kind"], f"{float(r['period_d']):.4f}") != key] + [jsonable(row)]
    tmp = path.with_suffix(".part")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def build(tic, signal: dict, cache_dir=None, lc_files=None, tpf_sectors=None, tpf_files=None, out_root=None,
          claimed_sectors=None, offline=False, neighbors=None) -> dict:
    tic = parse_tic(tic)
    cache_dir = Path(cache_dir or default_cache_dir())
    P, T0, dur_h = float(signal["period_d"]), float(signal["epoch_btjd"]), float(signal["duration_h"])
    out_root = Path(out_root or cache_dir / "packets")
    d = out_root / f"TIC{tic}_P{P:.5f}"
    d.mkdir(parents=True, exist_ok=True)
    notes, sub = [], {}

    # 1. gate (also gives per-event depths for the zooms)
    g = gate(tic, P, T0, dur_h, signal.get("depth_ppm"), claimed_sectors, cache_dir, lc_files, d / "_gate")
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
    ap.add_argument("--claimed-sectors", type=lambda s: [int(x) for x in s.split(",") if x])
    ap.add_argument("--tpf-sectors", type=lambda s: [int(x) for x in s.split(",") if x])
    ap.add_argument("--tpf-file", action="append")
    ap.add_argument("--lc-file", action="append")
    ap.add_argument("--offline", action="store_true", help="skip catalog queries and TPF downloads")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--out", type=Path, help="packet root (default <cache>/packets)")
    a = ap.parse_args(argv)
    sig = load_signal(a.signals, a.rank, a.period, a.epoch, a.duration, a.depth)
    s = build(a.tic, sig, a.cache_dir, a.lc_file, a.tpf_sectors, a.tpf_file, a.out, a.claimed_sectors, a.offline)
    print(json.dumps(jsonable({"tic": s["tic"], "status": s["status"], "packet_dir": s["packet_dir"], "files": s["files"],
                               "gate": s["results"]["allsector_gate"]["verdict"],
                               "centroid": s["results"]["centroid_check"]["verdict"],
                               "catalog": s["results"]["catalog_veto"]["verdict"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
