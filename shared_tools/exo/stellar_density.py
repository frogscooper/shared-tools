"""stellar_density: mean stellar density from a transit fit, and a check against the TIC.

    rho_star = 3*pi / (G P^2) * (a/Rs)^3            (circular orbit, Mp << M*)

Everything is cgs: G = 6.674e-8 cm^3 g^-1 s^-2, P in seconds, result in g/cm^3.
If you have a in AU and R* in Rsun instead of a/Rs, the conversion is explicit:
    a/Rs = a_AU * (AU_cm / Rsun_cm) / R*_Rsun,  AU/Rsun = 215.032
(forgetting this factor, i.e. the (AU/Rsun)^3 ~ 9.94e6 term, is the classic bug).
The solar mean density is 1.40980 g/cm^3 (IAU nominal M_sun, R_sun).

    stellar-density 4.411937 --a-rs 9.8785 --mstar 1.28 --rstar 1.21
"""
from __future__ import annotations

import argparse
import json
import math

G_CGS = 6.67430e-8
AU_CM = 1.495978707e13
RSUN_CM = 6.957e10
MSUN_G = 1.98840987e33
DAY_S = 86400.0
RHO_SUN_CGS = MSUN_G / (4.0 / 3.0 * math.pi * RSUN_CM ** 3)   # 1.4098 g/cm^3
AU_PER_RSUN = AU_CM / RSUN_CM                                   # 215.032


def a_over_rs(a_au: float, rstar_rsun: float) -> float:
    return a_au * AU_PER_RSUN / rstar_rsun


def stellar_density(period_d: float, a_rs: float | None = None, a_rs_err: float | None = None,
                    a_au: float | None = None, rstar_rsun: float | None = None) -> dict:
    if a_rs is None:
        if a_au is None or rstar_rsun is None:
            raise ValueError("give a_rs, or a_au and rstar_rsun")
        a_rs = a_over_rs(a_au, rstar_rsun)
    p_s = period_d * DAY_S
    rho = 3 * math.pi / (G_CGS * p_s ** 2) * a_rs ** 3
    out = {"period_d": period_d, "a_rs": a_rs, "rho_cgs": rho, "rho_solar": rho / RHO_SUN_CGS,
           "formula": "3*pi/(G P^2) * (a/Rs)^3, cgs", "rho_sun_cgs": RHO_SUN_CGS}
    if a_rs_err:
        out["rho_cgs_err"] = rho * 3 * a_rs_err / a_rs
        out["rho_solar_err"] = out["rho_cgs_err"] / RHO_SUN_CGS
    return out


def a_rs_from_density(period_d: float, rho_solar: float) -> float:
    """Inverse of stellar_density: a/Rs = (G rho P^2 / 3 pi)^(1/3), cgs."""
    p_s = period_d * DAY_S
    return (G_CGS * rho_solar * RHO_SUN_CGS * p_s ** 2 / (3 * math.pi)) ** (1.0 / 3.0)


def expected_duration_h(period_d: float, rho_solar: float, k: float = 0.0, b: float = 0.0) -> float:
    """Total transit duration T14 (hours) for a circular orbit: P/pi * asin(sqrt((1+k)^2 - b^2) / (a/Rs))."""
    a_rs = a_rs_from_density(period_d, rho_solar)
    x = math.sqrt(max((1 + k) ** 2 - b ** 2, 0.0)) / a_rs
    return period_d * 24 / math.pi * math.asin(min(x, 1.0))


def catalog_density_solar(mstar_msun: float, rstar_rsun: float) -> float:
    return mstar_msun / rstar_rsun ** 3


def compare(period_d, a_rs, mstar_msun, rstar_rsun, a_rs_err=None) -> dict:
    d = stellar_density(period_d, a_rs, a_rs_err)
    cat = catalog_density_solar(mstar_msun, rstar_rsun)
    d.update(catalog_rho_solar=cat, ratio_transit_over_catalog=d["rho_solar"] / cat)
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("period_d", type=float)
    ap.add_argument("--a-rs", type=float)
    ap.add_argument("--a-rs-err", type=float)
    ap.add_argument("--a-au", type=float)
    ap.add_argument("--mstar", type=float, help="Msun, for the catalog comparison")
    ap.add_argument("--rstar", type=float, help="Rsun (also used with --a-au)")
    a = ap.parse_args(argv)
    a_rs = a.a_rs if a.a_rs is not None else (a_over_rs(a.a_au, a.rstar) if a.a_au and a.rstar else None)
    if a.mstar and a.rstar and a_rs:
        r = compare(a.period_d, a_rs, a.mstar, a.rstar, a.a_rs_err)
    else:
        r = stellar_density(a.period_d, a_rs, a.a_rs_err, a.a_au, a.rstar)
    print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
