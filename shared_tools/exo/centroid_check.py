"""centroid_check: is the transit on the target star? Difference image + aperture growth.

For each target pixel file (TPF) in the requested sectors:
  diff = mean(out-of-transit frames) - mean(in-transit frames)
  - flux-weighted centroid of the positive diff pixels within 4 px of the target
  - brightest difference pixel
  - depth vs circular aperture radius (1 .. 4 px) and in the pipeline aperture
All pixel coordinates are 0-indexed TPF-array (column=x, row=y) from tpf.wcs.world_to_pixel.
TPF.column / TPF.row (the CCD origin) are deliberately never used: mixing them in was the
bug that put earlier centroids on the wrong pixel.

    centroid-check 149603524 --period 4.411937 --epoch 3982.065462 --duration 3.632 --sectors 1,2
    centroid-check 288183107 --period 12.96393 --epoch 3319.20707 --duration 6.208 --sectors 74

Saturated targets (brightest pixel > 8e4 e-/s, e.g. pi Men) bleed along columns, so for them the
verdict comes from the in/out-of-transit photocenter shift instead (reported for every TPF).

Flags per TPF: centroid_offset (windowed centroid > 1 px), peak_off_target (brightest
difference pixel > 2 px), depth_grows_with_aperture. A low-SNR difference image is "inconclusive".
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from .common import default_cache_dir, jsonable, now_et, parse_tic, write_json

PIX_ARCSEC = 21.0
WINDOW_PX = 4.0
RADII = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
CENTROID_MAX_PX = 1.0
PEAK_MAX_PX = 2.0
MIN_DIFF_SNR = 3.0
DEFAULT_BAD = 17087
SATURATION_E_S = 8.0e4   # TESS pixels saturate near 1e5 e-/s; above this, bleed columns smear the image


def target_pixel(tpf, ra: float, dec: float) -> tuple[float, float]:
    """Target (x=column, y=row) in 0-indexed TPF-array pixels. Uses only tpf.wcs."""
    from astropy.coordinates import SkyCoord

    x, y = tpf.wcs.world_to_pixel(SkyCoord(ra, dec, unit="deg"))
    return float(np.squeeze(x)), float(np.squeeze(y))


def propagate(ra, dec, pmra, pmdec, year, epoch=2000.0):
    """Move a TIC (epoch 2000) position by proper motion (mas/yr) to `year`."""
    if not pmra and not pmdec:
        return ra, dec
    dt = year - epoch
    dec2 = dec + (pmdec or 0) * dt / 3.6e6
    ra2 = ra + (pmra or 0) * dt / 3.6e6 / np.cos(np.radians(dec))
    return ra2, dec2


def btjd_to_year(t):
    return 2000.0 + (t + 2457000.0 - 2451545.0) / 365.25


def circle(shape, x, y, r):
    yy, xx = np.indices(shape)
    return (xx - x) ** 2 + (yy - y) ** 2 <= r * r


def transit_masks(t, good, tcs, dur_d, local_d=None):
    """In-transit (|dt| <= 0.45 dur) and out-of-transit (|dt| > 1.5 dur) cadences.

    local_d: keep only out-of-transit cadences within local_d of an event. Use it for single events,
    where a whole-sector baseline lets slow drifts and pointing changes dominate the difference image.
    """
    intr = np.zeros(len(t), bool)
    oot = np.ones(len(t), bool)
    near = np.zeros(len(t), bool)
    for tc in tcs:
        intr |= np.abs(t - tc) <= 0.45 * dur_d
        oot &= np.abs(t - tc) > 1.5 * dur_d
        if local_d:
            near |= np.abs(t - tc) <= local_d
    if local_d:
        oot &= near
    return intr & good, oot & good


def aperture_depth(t, flux, good, mask, tcs, dur_d):
    """Depth (ppm) in a pixel mask; local baseline 1.5-3 durations around each event."""
    if mask is None or mask.sum() == 0:
        return None
    sap = np.nansum(flux[:, mask], axis=1)
    ok = good & np.isfinite(sap) & (sap > 0)
    d_all, w_all = [], []
    for tc in tcs:
        dt = t - tc
        inn = ok & (np.abs(dt) <= 0.45 * dur_d)
        out = ok & (np.abs(dt) > 1.5 * dur_d) & (np.abs(dt) < 3.0 * dur_d)
        if inn.sum() < 2 or out.sum() < 5:
            continue
        # linear local baseline removes slow drifts across the event
        c = np.polyfit(dt[out], sap[out], 1)
        resid = sap[out] / np.polyval(c, dt[out])
        d = 1 - np.mean(sap[inn] / np.polyval(c, dt[inn]))
        e = np.std(resid) * np.sqrt(1 / inn.sum() + 1 / out.sum())
        d_all.append(d)
        w_all.append(1 / e ** 2)
    if not d_all:
        return None
    w = np.array(w_all)
    return {"depth_ppm": round(float((np.array(d_all) * w).sum() / w.sum()) * 1e6, 1),
            "err_ppm": round(float(1 / np.sqrt(w.sum())) * 1e6, 1), "n_pix": int(mask.sum()), "n_events": len(d_all)}


def photocenter_shift(t, flux, good, mask, tcs, dur_d, depth_frac, x0, y0) -> dict:
    """In- vs out-of-transit photocenter shift in the pipeline aperture -> implied source position.

    For a transit on a source at x_src, the aperture photocenter moves by
    dx = -depth/(1-depth) * (x_src - x_c), so x_src = x_c - dx * (1-depth)/depth.
    This still works for saturated stars, where the difference-image peak sits at a bleed-column end.
    """
    if mask is None or not mask.any() or not depth_frac or depth_frac <= 0:
        return {"status": "unavailable"}
    yy, xx = np.indices(mask.shape)
    f = np.where(mask, flux, 0.0)
    f = np.nan_to_num(f)
    tot = f.sum(axis=(1, 2))
    ok = good & (tot > 0)
    cx = (f * xx).sum(axis=(1, 2)) / np.where(tot > 0, tot, 1)
    cy = (f * yy).sum(axis=(1, 2)) / np.where(tot > 0, tot, 1)
    shifts = {"x": [], "y": []}
    errs = {"x": [], "y": []}
    for tc in tcs:
        dt = t - tc
        inn = ok & (np.abs(dt) <= 0.45 * dur_d)
        out = ok & (np.abs(dt) > 1.5 * dur_d) & (np.abs(dt) < 3.0 * dur_d)
        if inn.sum() < 2 or out.sum() < 5:
            continue
        for k, c in (("x", cx), ("y", cy)):
            p = np.polyfit(dt[out], c[out], 1)
            shifts[k].append(np.mean(c[inn] - np.polyval(p, dt[inn])))
            errs[k].append(np.std(c[out] - np.polyval(p, dt[out])) * np.sqrt(1 / inn.sum() + 1 / out.sum()))
    if not shifts["x"]:
        return {"status": "too_few_events"}
    res = {}
    for k in ("x", "y"):
        w = 1 / np.array(errs[k]) ** 2
        res[k] = float((np.array(shifts[k]) * w).sum() / w.sum())
        res[k + "_err"] = float(1 / np.sqrt(w.sum()))
    xc, yc = float(np.mean(cx[ok])), float(np.mean(cy[ok]))
    g = (1 - depth_frac) / depth_frac
    sx, sy = xc - res["x"] * g, yc - res["y"] * g
    off = float(np.hypot(sx - x0, sy - y0))
    sig = float(np.hypot(res["x_err"], res["y_err"]) * g)
    verdict = offset_verdict(off, sig)
    return {"status": "ok", "verdict": verdict, "source_dx_px": sx - x0, "source_dy_px": sy - y0,
            "source_dx_err_px": res["x_err"] * g, "source_dy_err_px": res["y_err"] * g, "shift_x_px": res["x"], "shift_y_px": res["y"],
            "shift_err_px": float(np.hypot(res["x_err"], res["y_err"])),
            "source_x": round(sx, 3), "source_y": round(sy, 3), "source_offset_px": round(off, 3),
            "source_offset_err_px": round(sig, 3), "photocenter_x": round(xc, 3), "photocenter_y": round(yc, 3)}


def offset_verdict(off, sig):
    if sig > CENTROID_MAX_PX:
        return "inconclusive"
    return "off_target" if off - 3 * sig > CENTROID_MAX_PX else "on_target"


def combine_photocenter(rows) -> dict:
    """Weighted mean of per-sector implied source offsets (pixel offsets from the target)."""
    pcs = [r["photocenter"] for r in rows if r.get("photocenter", {}).get("status") == "ok"]
    if not pcs:
        return {"status": "unavailable"}
    out = {}
    for k in ("dx", "dy"):
        v = np.array([p[f"source_{k}_px"] for p in pcs])
        w = 1 / np.array([p[f"source_{k}_err_px"] for p in pcs]) ** 2
        out[k], out[k + "_err"] = float((v * w).sum() / w.sum()), float(1 / np.sqrt(w.sum()))
    off = float(np.hypot(out["dx"], out["dy"]))
    sig = float(np.hypot(out["dx_err"], out["dy_err"]))
    return {"status": "ok", "n_sectors": len(pcs), "source_dx_px": round(out["dx"], 3), "source_dy_px": round(out["dy"], 3),
            "source_offset_px": round(off, 3), "source_offset_err_px": round(sig, 3), "verdict": offset_verdict(off, sig)}


def analyze_tpf(tpf, ra, dec, period_d, epoch_btjd, duration_h, pmra=None, pmdec=None, oot_local_d=None) -> tuple[dict, dict]:
    """Return (result row, images for plotting) for one TPF."""
    dur_d = duration_h / 24.0
    t = np.asarray(tpf.time.value, float)
    flux = np.asarray(getattr(tpf.flux, "value", tpf.flux), float)
    q = np.asarray(tpf.quality, int)
    sector = int(tpf.meta.get("SECTOR", -1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fin = np.isfinite(t) & np.isfinite(np.nanmean(flux, axis=(1, 2)))
    good = fin & ((q & DEFAULT_BAD) == 0)
    n = np.arange(np.floor((np.nanmin(t[fin]) - epoch_btjd) / period_d), np.ceil((np.nanmax(t[fin]) - epoch_btjd) / period_d) + 1)
    tcs = [tc for tc in epoch_btjd + n * period_d if np.any(good & (np.abs(t - tc) <= 0.45 * dur_d))]
    ra_e, dec_e = propagate(ra, dec, pmra, pmdec, btjd_to_year(np.nanmedian(t[fin])))
    x0, y0 = target_pixel(tpf, ra_e, dec_e)
    row = {"sector": sector, "author": str(tpf.meta.get("AUTHOR", "") or tpf.meta.get("PROCVER", "")),
           "shape": list(flux.shape[1:]), "target_x": round(x0, 3), "target_y": round(y0, 3),
           "coords": "0-indexed TPF array pixels from tpf.wcs.world_to_pixel (proper motion applied)",
           "n_events": len(tcs), "events_btjd": [round(float(x), 4) for x in tcs]}
    if not tcs:
        row.update(status="no_transit_in_tpf", verdict="inconclusive")
        return row, {}
    intr, oot = transit_masks(t, good, tcs, dur_d, oot_local_d)
    row.update(n_in=int(intr.sum()), n_out=int(oot.sum()), oot_local_d=oot_local_d)
    if intr.sum() < 3 or oot.sum() < 10:
        row.update(status="too_few_cadences", verdict="inconclusive")
        return row, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_img = np.nanmean(flux[oot], axis=0)
        in_img = np.nanmean(flux[intr], axis=0)
        noise = np.nanstd(flux[oot], axis=0) * np.sqrt(1 / intr.sum() + 1 / oot.sum())
    diff = out_img - in_img
    yy, xx = np.indices(diff.shape)
    pr, pc = np.unravel_index(int(np.nanargmax(np.where(np.isfinite(diff), diff, -np.inf))), diff.shape)
    pos = np.where(np.isfinite(diff), np.clip(diff, 0, None), 0.0) * circle(diff.shape, x0, y0, WINDOW_PX)
    if pos.sum() > 0:
        cx, cy = float((pos * xx).sum() / pos.sum()), float((pos * yy).sum() / pos.sum())
    else:
        cx = cy = float("nan")
    ix, iy = int(np.clip(round(x0), 0, diff.shape[1] - 1)), int(np.clip(round(y0), 0, diff.shape[0] - 1))
    snr_t = float(diff[iy, ix] / noise[iy, ix]) if noise[iy, ix] > 0 else None
    snr_p = float(diff[pr, pc] / noise[pr, pc]) if noise[pr, pc] > 0 else None
    off_c = float(np.hypot(cx - x0, cy - y0)) if np.isfinite(cx) else None
    off_p = float(np.hypot(pc - x0, pr - y0))
    pipe = getattr(tpf, "pipeline_mask", None)
    radii = []
    for r in RADII:
        d = aperture_depth(t, flux, good, circle(diff.shape, x0, y0, r), tcs, dur_d)
        if d:
            radii.append({"radius_px": r, **d})
    pipe_d = aperture_depth(t, flux, good, pipe, tcs, dur_d) if pipe is not None and pipe.any() else None
    grows = False
    if len(radii) >= 2:
        a, b = radii[0], radii[-1]
        grows = b["depth_ppm"] > 1.5 * a["depth_ppm"] and (b["depth_ppm"] - a["depth_ppm"]) > 3 * np.hypot(a["err_ppm"], b["err_ppm"])
    saturated = bool(np.nanmax(out_img) > SATURATION_E_S)
    depth_frac = (pipe_d or (radii[-1] if radii else {}) or {}).get("depth_ppm")
    pc_shift = photocenter_shift(t, flux, good, pipe if pipe is not None and pipe.any() else circle(diff.shape, x0, y0, 3),
                                 tcs, dur_d, None if depth_frac is None else depth_frac * 1e-6, x0, y0)
    flags = {"saturated": saturated,
             "no_dip_at_target": snr_t is not None and snr_t < 0,
             "centroid_offset": off_c is not None and off_c > CENTROID_MAX_PX,
             "peak_off_target": off_p > PEAK_MAX_PX,
             "depth_grows_with_aperture": bool(grows),
             "low_diff_snr": max(snr_t or 0, snr_p or 0) < MIN_DIFF_SNR}
    if saturated:
        # Bleed columns make the difference-image peak and aperture growth meaningless; trust the photocenter.
        method = "photocenter_shift"
        verdict = pc_shift.get("verdict", "inconclusive")
    elif flags["low_diff_snr"]:
        method, verdict = "difference_image", "inconclusive"
    elif flags["centroid_offset"] or flags["peak_off_target"] or flags["depth_grows_with_aperture"]:
        method, verdict = "difference_image", "off_target"
    else:
        method, verdict = "difference_image", "on_target"
    row.update(status="ok", verdict=verdict, method=method, flags=flags, photocenter=pc_shift,
               centroid_x=round(cx, 3), centroid_y=round(cy, 3),
               centroid_offset_px=None if off_c is None else round(off_c, 3),
               centroid_offset_arcsec=None if off_c is None else round(off_c * PIX_ARCSEC, 1),
               peak_x=int(pc), peak_y=int(pr), peak_offset_px=round(off_p, 3),
               peak_offset_arcsec=round(off_p * PIX_ARCSEC, 1),
               diff_snr_at_target=None if snr_t is None else round(snr_t, 2),
               diff_snr_at_peak=None if snr_p is None else round(snr_p, 2),
               depth_vs_radius=radii, depth_pipeline_aperture=pipe_d)
    return row, {"out": out_img, "diff": diff, "pipe": pipe}


def plot_tpf(row, img, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(np.log10(np.clip(img["out"], 1, None)), origin="lower", cmap="gray")
    ax[0].set_title(f"S{row['sector']} out-of-transit (log)")
    im = ax[1].imshow(img["diff"], origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    ax[1].set_title(f"difference (out - in): {row['verdict']}")
    for a in ax[:2]:
        a.plot(row["target_x"], row["target_y"], "rx", ms=10, mew=2, label="target (WCS)")
        a.plot(row["centroid_x"], row["centroid_y"], "w+", ms=12, mew=2, label="4-px centroid")
        a.plot(row["peak_x"], row["peak_y"], "s", mfc="none", mec="orange", ms=10, mew=2, label="brightest diff px")
        if img["pipe"] is not None:
            a.contour(img["pipe"], levels=[0.5], colors="cyan", linewidths=0.8)
    ax[1].legend(fontsize=7, loc="upper right")
    r = row["depth_vs_radius"]
    if r:
        ax[2].errorbar([x["radius_px"] for x in r], [x["depth_ppm"] for x in r], [x["err_ppm"] for x in r], fmt="o-")
    if row.get("depth_pipeline_aperture"):
        ax[2].axhline(row["depth_pipeline_aperture"]["depth_ppm"], color="cyan", ls="--", label="pipeline aperture")
        ax[2].legend(fontsize=7)
    ax[2].set(xlabel="aperture radius [px]", ylabel="depth [ppm]",
              title=f"centroid {row['centroid_offset_px']} px, peak {row['peak_offset_px']} px")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def check(tic, period_d, epoch_btjd, duration_h, sectors=None, ra=None, dec=None, pmra=None, pmdec=None,
          cache_dir=None, tpf_files=None, out_dir=None, plot=True, authors=("SPOC", "TESS-SPOC"), oot_local_d=None) -> dict:
    import lightkurve as lk

    tic = parse_tic(tic)
    cache_dir = Path(cache_dir or default_cache_dir())
    pos_src = "given"
    if ra is None or dec is None:
        from .catalog_veto import tic_position

        p = tic_position(tic)
        ra, dec = p["ra"], p["dec"]
        pmra = p.get("pmra") if pmra is None else pmra
        pmdec = p.get("pmdec") if pmdec is None else pmdec
        pos_src = "MAST TIC (epoch 2000) + proper motion"
    load = {}
    if tpf_files:
        files = [str(f) for f in tpf_files]
    else:
        from .tess_lc_fetch import fetch

        _, m = fetch(tic, authors=authors, cache_dir=cache_dir, pick="best", load=False, sectors=sectors, kind="tpf")
        files = [p["local_path"] for p in m["products"]]
        load = {"manifest": m["manifest_paths"][0], "failures": m["failures"]}
    out_dir = Path(out_dir or cache_dir / "centroid_check" / f"TIC{tic}_P{float(period_d):.5f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tpf = lk.read(f, quality_bitmask="none")
        if sectors and int(tpf.meta.get("SECTOR", -1)) not in set(sectors):
            continue
        row, img = analyze_tpf(tpf, ra, dec, float(period_d), float(epoch_btjd), float(duration_h), pmra, pmdec, oot_local_d)
        row["file"] = f
        if plot and img:
            row["plot"] = str(plot_tpf(row, img, out_dir / f"centroid_s{row['sector']}.png"))
        rows.append(row)
    combined = combine_photocenter(rows)
    verdicts = [r["verdict"] for r in rows]
    if rows and all(r.get("flags", {}).get("saturated") for r in rows):
        # every TPF saturated: the multi-sector photocenter is the only meaningful test
        verdict, basis = combined.get("verdict", "inconclusive"), "combined_photocenter (saturated target)"
    elif "off_target" in verdicts:
        verdict, basis = "off_target", "per-TPF difference image"
    elif "on_target" in verdicts:
        verdict, basis = "on_target", "per-TPF difference image"
    else:
        verdict, basis = "inconclusive", "no TPF gave a usable answer"
    result = {"tool": "centroid_check", "tic": tic, "generated_at": now_et(),
              "inputs": {"period_d": period_d, "epoch_btjd": epoch_btjd, "duration_h": duration_h, "sectors": sectors,
                         "ra": ra, "dec": dec, "pmra": pmra, "pmdec": pmdec, "position_source": pos_src},
              "thresholds": {"window_px": WINDOW_PX, "centroid_max_px": CENTROID_MAX_PX, "peak_max_px": PEAK_MAX_PX,
                             "min_diff_snr": MIN_DIFF_SNR, "pixel_arcsec": PIX_ARCSEC},
              "verdict": verdict, "verdict_basis": basis, "combined_photocenter": combined, "tpfs": rows, "load": load}
    write_json(out_dir / "centroid.json", result)
    result["json"] = str(out_dir / "centroid.json")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tic")
    ap.add_argument("--period", type=float, required=True)
    ap.add_argument("--epoch", type=float, required=True, help="BTJD")
    ap.add_argument("--duration", type=float, required=True, help="hours")
    ap.add_argument("--sectors", type=lambda s: [int(x) for x in s.split(",") if x], help="claimed-detection sectors")
    ap.add_argument("--ra", type=float)
    ap.add_argument("--dec", type=float)
    ap.add_argument("--tpf-file", action="append")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args(argv)
    r = check(a.tic, a.period, a.epoch, a.duration, a.sectors, a.ra, a.dec, cache_dir=a.cache_dir,
              tpf_files=a.tpf_file, out_dir=a.out_dir)
    keep = ("sector", "verdict", "method", "target_x", "target_y", "centroid_offset_px", "peak_offset_px", "diff_snr_at_target",
            "flags", "photocenter", "n_events", "plot")
    print(json.dumps(jsonable({"tic": r["tic"], "verdict": r["verdict"], "verdict_basis": r["verdict_basis"],
                               "combined_photocenter": r["combined_photocenter"], "json": r["json"],
                               "tpfs": [{k: x.get(k) for k in keep} for x in r["tpfs"]]}), indent=2))
    return {"on_target": 0, "off_target": 1, "inconclusive": 3}[r["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
