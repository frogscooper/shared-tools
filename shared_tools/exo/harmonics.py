"""period_harmonic_flag: is a period suspiciously close to a TESS orbit harmonic?

TESS orbits every ~13.7 d. Scattered light, momentum dumps and the mid-sector
downlink gap repeat on that clock, so periods near 6.85, 13.7 or 27.4 d deserve
extra suspicion.

    period-harmonic-flag 13.622518
"""
from __future__ import annotations

import argparse
import json

TESS_HARMONICS_D = (6.85, 13.7, 27.4)


def period_harmonic_flag(period_d: float, tol: float = 0.05, refs=TESS_HARMONICS_D) -> dict:
    """Offset (days) and percent offset of period_d from each reference period.

    flagged is True when |period - ref| / ref <= tol for any ref.
    """
    period_d = float(period_d)
    rows = []
    for ref in refs:
        frac = (period_d - ref) / ref
        rows.append({
            "ref_d": ref,
            "offset_d": round(period_d - ref, 6),
            "offset_pct": round(100 * frac, 3),
            "flagged": abs(frac) <= tol,
        })
    hits = [r["ref_d"] for r in rows if r["flagged"]]
    return {"period_d": period_d, "tol": tol, "flagged": bool(hits), "near": hits, "refs": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("period_d", type=float, nargs="+")
    ap.add_argument("--tol", type=float, default=0.05, help="fractional tolerance (default 0.05 = 5%%)")
    args = ap.parse_args(argv)
    out = [period_harmonic_flag(p, args.tol) for p in args.period_d]
    print(json.dumps(out[0] if len(out) == 1 else out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
