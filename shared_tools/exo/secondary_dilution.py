"""secondary_dilution: secondary eclipse search + which neighbors could fake the transit.

Secondary: depth and SNR at phase 0.5, plus a scan of every other phase (for eccentric
orbits) that reports the strongest non-primary dip. All on the detrended, quality-masked
light curves allsector_gate uses, with the primary transits masked from the trend.

Dilution: for every Gaia DR3 neighbor in the catalog_veto table,
    flux_fraction  = F_nb / (F_target + F_nb) = 1 / (1 + 10**(0.4 * delta_G))
    needed_depth   = observed_depth / flux_fraction
A neighbor "could_cause" the signal if needed_depth <= max_eclipse_depth (default 1.0 = total
eclipse) and it sits within max_sep_arcsec (default 63" = 3 TESS pixels). This is conservative:
it assumes all the neighbor's light lands in the aperture.

    secondary-dilution 149603524 --period 4.411937 --epoch 3982.065462 --duration 3.632 --depth 13200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .allsector_gate import load_sectors, prepare_sectors
from .folding import phase_depth
from .common import default_cache_dir, jsonable, now_et, parse_tic, write_json

SIGNIFICANT_SNR = 3.0
MAX_SEP_ARCSEC = 63.0


def flux_fraction(delta_g: float) -> float:
    return 1.0 / (1.0 + 10 ** (0.4 * delta_g))


def needed_depth(depth_ppm: float, delta_g: float) -> float:
    """Eclipse depth (fraction of the neighbor's own flux) needed to produce depth_ppm in the blend."""
    return depth_ppm * 1e-6 / flux_fraction(delta_g)


def dilution_table(neighbors, depth_ppm, max_sep_arcsec=MAX_SEP_ARCSEC, max_eclipse_depth=1.0) -> list[dict]:
    rows = []
    for n in neighbors:
        if n.get("is_target") or n.get("delta_g") is None:
            continue
        nd = needed_depth(depth_ppm, n["delta_g"])
        rows.append({"source_id": n.get("source_id"), "sep_arcsec": n.get("sep_arcsec"), "gmag": n.get("gmag"),
                     "delta_g": n["delta_g"], "flux_fraction": round(flux_fraction(n["delta_g"]), 6),
                     "needed_eclipse_depth": round(nd, 5),
                     "could_cause": bool(nd <= max_eclipse_depth and (n.get("sep_arcsec") or 0) <= max_sep_arcsec)})
    return sorted(rows, key=lambda r: r["needed_eclipse_depth"])


def secondary_search(t, f, t0, period, dur_d, step=None) -> dict:
    p05 = phase_depth(t, f, t0, period, dur_d, 0.5)
    prim = phase_depth(t, f, t0, period, dur_d, 0.0, exclude_primary=False)
    step = step or max(0.002, 0.25 * dur_d / period)
    margin = 2.0 * dur_d / period
    scan = [phase_depth(t, f, t0, period, dur_d, c) for c in np.arange(margin, 1 - margin, step)]
    scan = [s for s in scan if s["snr"] is not None]
    strongest = max(scan, key=lambda s: s["snr"]) if scan else None
    # Expected max SNR from pure noise over this many trials (~sqrt(2 ln N)), to judge the scan.
    n_ind = max(1, int((1 - 2 * margin) * period / (0.8 * dur_d)))
    return {"phase_0p5": p05, "primary": prim, "strongest_other_phase": strongest,
            "scan_trials_independent": n_ind, "scan_noise_expect_snr": round(float(np.sqrt(2 * np.log(max(n_ind, 2)))), 2),
            "significant_at_0p5": bool(p05["snr"] is not None and p05["snr"] >= SIGNIFICANT_SNR),
            "significant_elsewhere": bool(strongest is not None and strongest["snr"] >= max(SIGNIFICANT_SNR + 2,
                                          np.sqrt(2 * np.log(max(n_ind, 2))) + 2)),
            "scan": scan}


def run(tic, period_d, epoch_btjd, duration_h, depth_ppm=None, neighbors=None, cache_dir=None, lc_files=None,
        out_dir=None, max_sep_arcsec=MAX_SEP_ARCSEC, plot=True) -> dict:
    tic = parse_tic(tic)
    P, T0, dur_d = float(period_d), float(epoch_btjd), float(duration_h) / 24
    secs, load = load_sectors(tic, cache_dir, lc_files)
    prepare_sectors(secs, P, T0, dur_d)
    t = np.concatenate([s["time"][s["good"]] for s in secs])
    f = np.concatenate([s["flat"][s["good"]] for s in secs])
    ok = np.isfinite(f)
    t, f = t[ok], f[ok]
    sec = secondary_search(t, f, T0, P, dur_d)
    obs = depth_ppm or sec["primary"]["depth_ppm"]
    nb_source = "given"
    if neighbors is None:
        from .catalog_veto import veto

        v = veto(tic, cache_dir=cache_dir)
        neighbors = v["sources"].get("gaia_dr3_neighbors", {}).get("matches", [])
        nb_source = f"catalog_veto (gaia_dr3_neighbors status {v['sources'].get('gaia_dr3_neighbors', {}).get('status')})"
    dil = dilution_table(neighbors, obs, max_sep_arcsec) if obs else []
    ratio = (sec["phase_0p5"]["depth_ppm"] / obs) if (obs and sec["phase_0p5"]["depth_ppm"] is not None) else None
    result = {
        "tool": "secondary_dilution", "tic": tic, "generated_at": now_et(),
        "inputs": {"period_d": P, "epoch_btjd": T0, "duration_h": duration_h, "depth_ppm": depth_ppm,
                   "max_sep_arcsec": max_sep_arcsec},
        "observed_depth_ppm": obs,
        "secondary": {k: v for k, v in sec.items() if k != "scan"},
        "secondary_to_primary_ratio": None if ratio is None else round(ratio, 4),
        "dilution": {"neighbor_source": nb_source, "n_neighbors": len(dil),
                     "n_could_cause": sum(r["could_cause"] for r in dil), "could_cause": [r for r in dil if r["could_cause"]],
                     "table": dil},
        "flags": {"secondary_at_0p5": sec["significant_at_0p5"], "dip_elsewhere_in_phase": sec["significant_elsewhere"],
                  "neighbor_could_cause": any(r["could_cause"] for r in dil)},
        "load": load,
    }
    out_dir = Path(out_dir or Path(cache_dir or default_cache_dir()) / "secondary_dilution" / f"TIC{tic}_P{P:.5f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if plot:
        result["plot"] = str(make_plot(result, sec["scan"], out_dir / "secondary.png"))
    write_json(out_dir / "secondary_dilution.json", result)
    result["json"] = str(out_dir / "secondary_dilution.json")
    return result


def make_plot(res, scan, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    if scan:
        ax[0].errorbar([s["phase"] for s in scan], [s["depth_ppm"] for s in scan], [s["err_ppm"] for s in scan],
                       fmt=".", ms=3, elinewidth=0.5)
    ax[0].axvline(0.5, color="r", ls=":")
    ax[0].axhline(0, color="gray", lw=0.5)
    p = res["secondary"]["phase_0p5"]
    ax[0].set(xlabel="phase", ylabel="box depth [ppm]", title=f"TIC {res['tic']} phase scan; 0.5: {p['depth_ppm']} ppm, SNR {p['snr']}")
    tab = res["dilution"]["table"]
    if tab:
        c = ["red" if r["could_cause"] else "gray" for r in tab]
        ax[1].scatter([r["sep_arcsec"] for r in tab], [r["needed_eclipse_depth"] for r in tab], c=c, s=12)
        ax[1].set_yscale("log")
        ax[1].axhline(1.0, color="k", ls="--", lw=0.8)
        ax[1].axvline(res["inputs"]["max_sep_arcsec"], color="k", ls=":", lw=0.8)
    ax[1].set(xlabel="separation [arcsec]", ylabel="needed eclipse depth", title="neighbors (red = could cause)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tic")
    ap.add_argument("--period", type=float, required=True)
    ap.add_argument("--epoch", type=float, required=True, help="BTJD")
    ap.add_argument("--duration", type=float, required=True, help="hours")
    ap.add_argument("--depth", type=float, help="observed depth, ppm (default: measured primary fold)")
    ap.add_argument("--veto-json", type=Path, help="catalog_veto result.json to take neighbors from")
    ap.add_argument("--lc-file", action="append")
    ap.add_argument("--max-sep", type=float, default=MAX_SEP_ARCSEC)
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args(argv)
    nb = None
    if a.veto_json:
        nb = json.loads(a.veto_json.read_text())["sources"]["gaia_dr3_neighbors"]["matches"]
    r = run(a.tic, a.period, a.epoch, a.duration, a.depth, nb, a.cache_dir, a.lc_file, a.out_dir, a.max_sep)
    print(json.dumps(jsonable({k: r[k] for k in ("tic", "observed_depth_ppm", "secondary", "secondary_to_primary_ratio",
                                                 "flags", "json")} | {"could_cause": r["dilution"]["could_cause"][:10],
                                                                      "n_neighbors": r["dilution"]["n_neighbors"]}), indent=2))
    return 1 if any(r["flags"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
