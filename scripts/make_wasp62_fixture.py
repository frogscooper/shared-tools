"""Build tests/fixtures/wasp62_s1_s2_10min.csv.gz from the SPOC 2-min light curves.

WASP-62 b (TIC 149603524) is a known hot Jupiter used only as a control. Each sector is
binned to 10 minutes so CI can run the all-sector gate offline on a ~100 KB file.

    python scripts/make_wasp62_fixture.py /path/to/s0001_lc.fits /path/to/s0002_lc.fits
"""
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

DEFAULT_BAD, DESAT = 17087, 32
BIN_D = 10 / 1440


def bin_sector(path):
    with fits.open(path) as h:
        sector, d = h[0].header["SECTOR"], h[1].data
        t, f, e, q = (np.asarray(d[c], float) for c in ("TIME", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR", "QUALITY"))
    q = q.astype(int)
    good = np.isfinite(t) & np.isfinite(f) & ((q & DEFAULT_BAD) == 0)
    med = np.nanmedian(f[good])
    idx = np.floor((t - np.nanmin(t)) / BIN_D)
    rows = []
    for k in np.unique(idx[np.isfinite(idx)]):
        allin = idx == k
        sel = allin & good
        if sel.sum() < 3:
            continue
        bits = int(np.bitwise_or.reduce(q[allin]))
        rows.append({"sector": sector, "author": "SPOC", "cadence_s": 600.0,
                     "time": round(float(t[sel].mean()), 6), "flux": round(float(f[sel].mean() / med), 7),
                     "flux_err": round(float(np.sqrt((e[sel] ** 2).sum()) / sel.sum() / med), 7),
                     # keep informational bits (stray light etc.) and momentum dumps; drop single bad cadences
                     "quality": (bits & ~DEFAULT_BAD) | (bits & DESAT)})
    return rows


if __name__ == "__main__":
    rows = [r for p in sys.argv[1:] for r in bin_sector(p)]
    out = "tests/fixtures/wasp62_s1_s2_10min.csv.gz"
    pd.DataFrame(rows).to_csv(out, index=False, compression="gzip")
    print(out, len(rows), "rows")
