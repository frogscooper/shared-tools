"""duration_check: is a transit's duration possible for this star, period and depth?

For a circular orbit, the total duration is

    T14 = P/pi * asin( sqrt((1+k)^2 - b^2) / (a/Rs) ),   a/Rs from rho* and P (stellar_density)

with k = Rp/R* ~ sqrt(depth). Given the observed T14 we solve for the impact parameter b:

    b^2 = (1+k)^2 - (a/Rs * sin(pi T14 / P))^2

and also for the stellar density that a central (b = 0) transit would need:

    rho_b0 = 3 pi / (G P^2) * ((1+k) / sin(pi T14 / P))^3

Flags (a "fail" here means "a planet around this star on a circular orbit is hard to make"):

  too_short  the observed T14 needs b > 1 - k (a grazing transit). A grazing body this deep is
             usually an eclipsing binary, and k from sqrt(depth) is then only a lower limit.
  too_long   T14 is more than `long_factor` (1.5) times the b = 0 maximum: the host is less dense
             than the TIC says (evolved, or the wrong star in a blend), or the orbit is eccentric.

BLS box widths sit between T23 and T14 and come from a coarse grid, so the defaults are
deliberately loose. A pass is not evidence of a planet.

    duration-check 10.4317 --t14 1.2 --depth 1370 --rho 0.78
    duration-check 10.4317 --t14 1.2 --depth 1370 --mstar 0.9 --rstar 0.97
"""
from __future__ import annotations

import argparse
import json
import math

from .stellar_density import G_CGS, DAY_S, RHO_SUN_CGS, a_rs_from_density, catalog_density_solar, expected_duration_h

LONG_FACTOR = 1.5
GRAZE_MARGIN = 0.0   # b must exceed 1 - k + margin to call it grazing


def check(period_d: float, t14_h: float, depth_ppm: float, rho_solar: float | None = None,
          mstar: float | None = None, rstar: float | None = None, long_factor: float = LONG_FACTOR) -> dict:
    if rho_solar is None and mstar and rstar:
        rho_solar = catalog_density_solar(mstar, rstar)
    k = math.sqrt(max(float(depth_ppm), 0.0) * 1e-6)
    s = math.sin(math.pi * (t14_h / 24.0) / period_d)
    rho_b0_solar = 3 * math.pi / (G_CGS * (period_d * DAY_S) ** 2) * ((1 + k) / s) ** 3 / RHO_SUN_CGS if s > 0 else None
    out = {"tool": "duration_check", "period_d": period_d, "t14_obs_h": t14_h, "depth_ppm": depth_ppm, "k": round(k, 4),
           "rho_star_solar": rho_solar, "rho_needed_for_b0_solar": None if rho_b0_solar is None else round(rho_b0_solar, 3)}
    if not rho_solar or not math.isfinite(rho_solar) or rho_solar <= 0:
        out.update(result="incomplete", reason="no stellar density")
        return out
    a_rs = a_rs_from_density(period_d, rho_solar)
    t14_b0 = expected_duration_h(period_d, rho_solar, k, 0.0)
    b2 = (1 + k) ** 2 - (a_rs * s) ** 2
    b = math.sqrt(b2) if b2 > 0 else None          # None: longer than any b allows
    ratio = t14_h / t14_b0
    out.update(a_rs=round(a_rs, 3), t14_b0_h=round(t14_b0, 3), ratio_obs_over_b0=round(ratio, 3),
               implied_b=None if b is None else round(b, 3), grazing_limit_b=round(1 - k, 3),
               rho_ratio_needed_over_star=None if rho_b0_solar is None else round(rho_b0_solar / rho_solar, 2))
    if b is not None and b > 1 - k + GRAZE_MARGIN:
        out.update(result="fail", flag="too_short",
                   reason=f"T14 {t14_h:.2f} h vs {t14_b0:.2f} h at b=0 needs b={b:.2f} > 1-k={1 - k:.2f} (grazing); "
                          f"a central transit needs rho*={rho_b0_solar:.2f} solar vs {rho_solar:.2f}")
    elif ratio > long_factor:
        out.update(result="fail", flag="too_long",
                   reason=f"T14 {t14_h:.2f} h is {ratio:.1f}x the b=0 maximum {t14_b0:.2f} h "
                          f"(host less dense than TIC, blend, or eccentric)")
    else:
        out.update(result="pass", flag=None, reason=f"T14/T14(b=0) = {ratio:.2f}, b ~ {b if b is not None else 0:.2f}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("period_d", type=float)
    ap.add_argument("--t14", type=float, required=True, help="observed total duration, hours")
    ap.add_argument("--depth", type=float, required=True, help="ppm")
    ap.add_argument("--rho", type=float, help="stellar density, solar units (TIC 'rho')")
    ap.add_argument("--mstar", type=float)
    ap.add_argument("--rstar", type=float)
    ap.add_argument("--long-factor", type=float, default=LONG_FACTOR)
    a = ap.parse_args(argv)
    print(json.dumps(check(a.period_d, a.t14, a.depth, a.rho, a.mstar, a.rstar, a.long_factor), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
