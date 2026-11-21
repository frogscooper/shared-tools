"""masks: one reusable cadence mask for TESS light curves.

    good, why = cadence_mask(time, quality, author="SPOC")

Drops, and reports separately in `why`:
  quality       lightkurve "default" bad bits (17087); for QLP any nonzero flag
  momentum_dump Desat bit (32), padded by dump_pad_d on each side
  scattered     Straylight / Straylight2 bits (2048 | 4096)
  orbit_edge    the first and last edge_d days of every contiguous stretch (orbit or sector),
                where ramps from heating and scattered light live
  nonfinite     NaN time or flux
"""
from __future__ import annotations

import numpy as np

DEFAULT_BAD = 17087
DESAT = 32
SCATTERED = 2048 | 4096
EDGE_D = 1.0
GAP_D = 0.5          # a gap longer than this starts a new stretch (orbit gap, downlink, safe mode)
DUMP_PAD_D = 0.05


def stretches(time, gap_d=GAP_D):
    """(start, end) of contiguous stretches of finite time, split at gaps > gap_d."""
    t = np.sort(np.asarray(time, float)[np.isfinite(time)])
    if len(t) == 0:
        return []
    cut = np.where(np.diff(t) > gap_d)[0]
    return list(zip(np.r_[t[0], t[cut + 1]], np.r_[t[cut], t[-1]]))


def cadence_mask(time, quality, flux=None, author="SPOC", edge_d=EDGE_D, dump_pad_d=DUMP_PAD_D, gap_d=GAP_D,
                 bad_bits=DEFAULT_BAD, mask_scattered=True):
    """Return (good, why). good is a bool array; why maps reason -> bool array of dropped cadences."""
    t = np.asarray(time, float)
    q = np.asarray(quality, np.int64) if quality is not None else np.zeros(len(t), np.int64)
    fin = np.isfinite(t) & (np.isfinite(flux) if flux is not None else True)
    why = {"nonfinite": ~fin}
    why["quality"] = (q != 0) if str(author).upper() == "QLP" else (q & bad_bits) != 0
    dump = np.zeros(len(t), bool)
    for td in t[(q & DESAT) != 0]:
        dump |= np.abs(t - td) <= dump_pad_d
    why["momentum_dump"] = dump
    why["scattered"] = ((q & SCATTERED) != 0) if mask_scattered else np.zeros(len(t), bool)
    edge = np.zeros(len(t), bool)
    if edge_d and edge_d > 0:
        for a, b in stretches(t[fin], gap_d):
            edge |= ((t >= a) & (t < a + edge_d)) | ((t > b - edge_d) & (t <= b))
    why["orbit_edge"] = edge
    bad = np.zeros(len(t), bool)
    for m in why.values():
        bad |= m
    return ~bad, why


def summarize(why) -> dict:
    """Counts of dropped cadences per reason (a cadence can count under several)."""
    return {k: int(v.sum()) for k, v in why.items()}
