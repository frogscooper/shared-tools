"""Robust box depth at a given orbital phase, shared by allsector_gate and secondary_dilution."""
from __future__ import annotations

import numpy as np

CLIP_SIGMA = 5.0


def robust_std(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) > 2 else np.nan


def phase_depth(t, f, t0, period, dur_d, phase, exclude_primary=True):
    """Box depth (width 0.8 * duration) centered on `phase`.

    The depth is measured per orbit and the per-orbit values are combined after a 5-sigma
    clip, so one bad orbit (a scattered-light ramp, a momentum dump) can't fake a
    significant dip. The error is the larger of the white-noise error and the
    orbit-to-orbit scatter / sqrt(n_orbits).
    """
    ph = ((t - t0) / period) % 1.0
    dphase = ((ph - phase + 0.5) % 1.0) - 0.5
    dprim = ((ph + 0.5) % 1.0) - 0.5
    half = 0.4 * dur_d / period
    inn = np.abs(dphase) <= half
    out = (np.abs(dphase) > 3 * half) & ((np.abs(dprim) > 3 * half) if exclude_primary else True)
    empty = {"phase": round(float(phase), 4), "depth_ppm": None, "err_ppm": None, "snr": None, "n_in": int(inn.sum())}
    if inn.sum() < 10 or out.sum() < 30:
        return empty
    base = np.median(f[out])
    sig = robust_std(f[out])
    orbit = np.floor((t - t0) / period - phase + 0.5).astype(int)
    d, w = [], []
    for k in np.unique(orbit[inn]):
        m = inn & (orbit == k)
        if m.sum() >= 3:
            d.append(base - np.mean(f[m]))
            w.append(m.sum())
    if not d:
        return empty
    d, w = np.array(d), np.array(w, float)
    n_clipped = 0
    if len(d) >= 4:
        s = robust_std(d * np.sqrt(w))  # per-point-normalized spread
        keep = np.abs((d - np.median(d)) * np.sqrt(w)) <= CLIP_SIGMA * max(s, sig)
        n_clipped = int((~keep).sum())
        d, w = d[keep], w[keep]
    depth = float((d * w).sum() / w.sum())
    err_white = sig / np.sqrt(w.sum())
    err_orbit = robust_std(d) / np.sqrt(len(d)) if len(d) >= 4 else 0.0
    err = float(max(err_white, err_orbit))
    return {"phase": round(float(phase), 4), "depth_ppm": round(depth * 1e6, 1), "err_ppm": round(err * 1e6, 1),
            "snr": round(depth / err, 2) if err > 0 else None, "n_in": int(w.sum()), "n_orbits": int(len(d)),
            "n_orbits_clipped": n_clipped}
