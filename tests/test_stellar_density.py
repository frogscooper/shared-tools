import pytest

from shared_tools.exo.stellar_density import AU_PER_RSUN, RHO_SUN_CGS, compare, stellar_density


def test_constants():
    assert RHO_SUN_CGS == pytest.approx(1.4098, rel=1e-4)
    assert AU_PER_RSUN == pytest.approx(215.032, rel=1e-5)


def test_earth_sun_gives_solar_density():
    # Earth: P = 365.25636 d, a = 1 AU around a 1 Rsun star -> rho = rho_sun (Kepler's third law)
    r = stellar_density(365.25636, a_au=1.0, rstar_rsun=1.0)
    assert r["rho_solar"] == pytest.approx(1.0, rel=2e-3)


def test_wasp62b_matches_vetter_corrected_ratio():
    # vetter's batman fit: a/Rs = 9.8785 +/- 0.0530, P = 4.411937 d; TIC M = 1.28, R = 1.21
    r = compare(4.411937227697335, 9.878547344809917, 1.28, 1.21, a_rs_err=0.05301831821301501)
    assert r["rho_solar"] == pytest.approx(0.6645, abs=5e-4)
    assert r["rho_cgs"] == pytest.approx(0.9368, abs=1e-3)
    assert r["ratio_transit_over_catalog"] == pytest.approx(0.92, abs=0.005)


def test_a_au_path_equals_a_rs_path():
    a = stellar_density(4.0, a_au=0.05, rstar_rsun=1.1)
    b = stellar_density(4.0, a_rs=0.05 * AU_PER_RSUN / 1.1)
    assert a["rho_cgs"] == pytest.approx(b["rho_cgs"], rel=1e-12)
