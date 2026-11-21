"""Synthetic TESS-like sectors for offline tests (no network)."""
import numpy as np

from shared_tools.exo.duo_solver import trapezoid


def make_sector(sector, start, transits=(), ndays=27.0, cad_min=30.0, noise=500e-6, seed=0, quality=None,
                author="TESS-SPOC"):
    """transits: iterable of (t0, depth, t14_d, ingress_frac). A 1-day orbit gap sits mid-sector."""
    rng = np.random.default_rng(seed + 1000 * sector)
    t = np.arange(start, start + ndays, cad_min / 1440)
    mid = start + ndays / 2
    t = t[(t < mid - 0.5) | (t > mid + 0.5)]
    f = 1 + rng.normal(0, noise, len(t))
    for t0, depth, t14, tf in transits:
        f *= trapezoid(t, t0, depth, t14, tf * t14)
    q = np.zeros(len(t), int) if quality is None else quality(t)
    return {"sector": sector, "author": author, "cadence_s": cad_min * 60, "time": t, "flux": f * 1000.0,
            "flux_err": np.full(len(t), noise * 1000.0), "quality": q, "source": "synthetic"}
