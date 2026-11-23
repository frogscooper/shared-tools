"""bls_triage, duration_check and the banded BLS grid (all offline)."""
import pytest

from shared_tools.exo import bls_triage as bt
from shared_tools.exo.duration_check import check
from shared_tools.exo.transit_search import duration_bands


def sig(P, snr, **kw):
    base = {"kind": "periodic", "period_d": P, "epoch_btjd": 3000.3, "duration_h": 3.0, "depth_ppm": 1000.0, "snr": snr,
            "n_transits": 6, "odd_even_sigma": 0.5, "harmonic_flag": False, "duration_at_grid_edge": False}
    return {**base, **kw}


def star(tic, sigs, rstar=1.0, rho=1.0, singles=(), cover=((2990.0, 3050.0),)):
    return {"tic": tic, "rstar": rstar, "rho_solar": rho, "signals": sigs, "coverage": [list(c) for c in cover],
            "singles": [{"epoch_btjd": t} for t in singles]}


def test_duration_check_flags_grazing_tic141618383_and_passes_pi_men_c():
    g = check(10.4317, 1.2, 1370, 0.78)            # TIC 141618383 (hunt 21 Nov): grazing, likely EB
    assert g["result"] == "fail" and g["flag"] == "too_short" and g["implied_b"] > 0.96
    assert check(6.26782, 2.95, 290, 0.83)["result"] == "pass"    # pi Men c
    assert check(10.0, 14.0, 1000, 1.0)["flag"] == "too_long"
    assert check(10.0, 3.0, 1000, None)["result"] == "incomplete"


def test_each_cut():
    rows = {}
    cases = {
        "clean": [sig(5.37, 20)],
        "snr": [sig(5.37, 8)],
        "n_transits": [sig(5.37, 20, n_transits=2)],
        "odd_even": [sig(5.37, 20, odd_even_sigma=4.0)],
        "dominance": [sig(5.37, 20), sig(2.11, 15)],
        "harmonic": [sig(13.7, 20, harmonic_flag=True)],
        "integer_day": [sig(4.01, 20)],
        "radius": [sig(5.37, 20, depth_ppm=60000)],
        "grid_edge": [sig(1.37, 20, duration_at_grid_edge=True, duration_h=2.0)],
        "duration": [sig(10.4317, 20, duration_h=1.2, depth_ppm=1370)],
    }
    for i, (name, sigs) in enumerate(cases.items()):
        rows[name] = bt.triage_star(star(i, sigs, rho=0.78 if name == "duration" else 1.0), [])[0]
    assert rows["clean"]["status"] == "survivor"
    for name, r in rows.items():
        if name != "clean":
            assert name in r["fails"].split(";"), (name, r["fails"])
    assert rows["snr"]["fails"] == "snr" and rows["dominance"]["fails"] == "dominance"
    assert rows["snr"]["status"] == "near_miss" and rows["dominance"]["status"] == "near_miss"
    assert rows["radius"]["status"] == "cut" and rows["duration"]["status"] == "cut"


def test_shared_epoch_systematic_is_found_and_cut():
    # 30 stars all dip near BTJD 3010.2 (a CCD-wide systematic); one star's "planet" is built on those epochs
    stars = [star(100 + i, [sig(7.31 + 0.01 * i, 12)], singles=[3010.2, 2995.0 + i]) for i in range(30)]
    stars.append(star(7, [sig(10.0 + 0.37, 25, epoch_btjd=3010.2)], singles=[3010.2],
                      cover=((3005.0, 3015.0), (3018.0, 3024.0))))
    r = bt.triage(stars)
    assert r["bad_windows"] and r["bad_windows"][0]["start_btjd"] <= 3010.2 <= r["bad_windows"][0]["end_btjd"]
    row = next(x for x in r["near_misses"] + r["survivors"] if x["tic"] == 7) if any(
        x["tic"] == 7 for x in r["near_misses"] + r["survivors"]) else None
    assert row is not None and "systematic" in row["fails"]


def test_duration_bands_reach_long_durations_but_respect_bls_limit():
    bands = duration_bands(0.5, 60)
    assert bands[0][0] == 0.5 and bands[-1][1] == 60
    for lo, hi, d in bands:
        assert (d < 0.5 * lo).all()
    assert max(bands[0][2]) * 24 < 6.0                     # P = 0.5 d: only short boxes
    assert max(bands[-1][2]) * 24 >= 10.0                  # long periods can now fit > 4 h
    assert duration_bands(1.0, 12)[0][2].min() * 24 == pytest.approx(1.0)
