"""centroid_check on synthetic target pixel files (offline) and real TPFs (live)."""
import inspect
import types

import numpy as np
import pytest
from astropy.wcs import WCS

from shared_tools.exo import centroid_check as cc

RA0, DEC0 = 120.0, -30.0
P, T0, DUR_H = 3.0, 1001.0, 3.0


def make_wcs(shape=(11, 11)):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [shape[1] / 2 + 0.5, shape[0] / 2 + 0.5]   # FITS 1-indexed -> array center (5, 5)
    w.wcs.cdelt = [-21 / 3600, 21 / 3600]
    return w


class FakeTPF:
    """Minimal stand-in for a lightkurve TPF. Touching .column or .row fails the test."""

    def __init__(self, flux, t, wcs, pipe):
        self.flux = flux
        self.time = types.SimpleNamespace(value=t)
        self.quality = np.zeros(len(t), int)
        self.wcs = wcs
        self.pipeline_mask = pipe
        self.meta = {"SECTOR": 7, "AUTHOR": "SPOC"}

    @property
    def column(self):
        raise AssertionError("centroid_check must not use tpf.column (CCD origin); use tpf.wcs only")

    @property
    def row(self):
        raise AssertionError("centroid_check must not use tpf.row (CCD origin); use tpf.wcs only")


def psf(shape, x, y, amp, s=0.9):
    yy, xx = np.indices(shape)
    return amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))


def synth_tpf(eclipse_on="target", depth=0.01, seed=0, shape=(11, 11)):
    rng = np.random.default_rng(seed)
    t = np.arange(1000.0, 1027.0, 10 / 1440)
    intr = np.abs(((t - T0 + 0.5 * P) % P) - 0.5 * P) <= DUR_H / 48
    tx, ty = 5.0, 5.0                        # where the WCS puts RA0/DEC0
    nx, ny = 9.0, 5.0                        # neighbor 4 px away, 3x fainter
    target = psf(shape, tx, ty, 3000.0)
    neigh = psf(shape, nx, ny, 1000.0)
    flux = np.empty((len(t),) + shape)
    for i in range(len(t)):
        a, b = 1.0, 1.0
        if intr[i]:
            if eclipse_on == "target":
                a -= depth
            else:
                b -= depth * 4               # bigger eclipse on the faint neighbor
        flux[i] = a * target + b * neigh + 50 + rng.normal(0, 3, shape)
    pipe = psf(shape, tx, ty, 1.0) > 0.3
    return FakeTPF(flux, t, make_wcs(shape), pipe)


def test_target_pixel_is_wcs_world_to_pixel_without_origin():
    tpf = synth_tpf()
    x, y = cc.target_pixel(tpf, RA0, DEC0)       # would raise if column/row were touched
    from astropy.coordinates import SkyCoord

    ex, ey = tpf.wcs.world_to_pixel(SkyCoord(RA0, DEC0, unit="deg"))
    assert (x, y) == (float(ex), float(ey)) and x == pytest.approx(5.0) and y == pytest.approx(5.0)


def test_source_never_mentions_ccd_origin():
    src = inspect.getsource(cc)
    for bad in ("tpf.column", "tpf.row", ".column -", ".row -"):
        assert bad not in src.replace('"tpf.column', "").replace("tpf.column / tpf.row", "").replace(
            "TPF.column / TPF.row", ""), bad


def test_transit_on_target_is_on_target():
    row, _ = cc.analyze_tpf(synth_tpf("target"), RA0, DEC0, P, T0, DUR_H)
    assert row["verdict"] == "on_target"
    assert row["centroid_offset_px"] < 0.3 and row["peak_offset_px"] < 1.0
    assert row["photocenter"]["verdict"] == "on_target"


def test_transit_on_neighbor_is_off_target():
    row, _ = cc.analyze_tpf(synth_tpf("neighbor"), RA0, DEC0, P, T0, DUR_H)
    assert row["verdict"] == "off_target"
    assert row["flags"]["peak_off_target"] and row["peak_offset_px"] == pytest.approx(4.0, abs=0.6)


def test_depth_vs_radius_present():
    row, _ = cc.analyze_tpf(synth_tpf("target"), RA0, DEC0, P, T0, DUR_H)
    radii = [r["radius_px"] for r in row["depth_vs_radius"]]
    assert radii[0] == 1.0 and radii[-1] == 4.0
    assert row["depth_vs_radius"][0]["depth_ppm"] == pytest.approx(1e4, rel=0.15)


def test_proper_motion_propagation():
    ra, dec = cc.propagate(10.0, 0.0, 3600.0, 3600.0, 2001.0)   # 3.6"/yr = 0.001 deg/yr
    assert ra == pytest.approx(10.001) and dec == pytest.approx(0.001)


@pytest.mark.live
def test_live_wasp62_on_target():
    r = cc.check(149603524, 4.411937227697335, 3982.065462, 3.632, sectors=[1, 2], plot=False)
    offs = {t["sector"]: t["centroid_offset_px"] for t in r["tpfs"]}
    assert r["verdict"] == "on_target" and all(o < 0.3 for o in offs.values())


@pytest.mark.live
def test_live_288183107_s74_peak_off():
    r = cc.check(288183107, 12.963929714582635, 3319.20706782588, 6.208457815172235, sectors=[74], plot=False)
    s74 = r["tpfs"][0]
    assert s74["peak_offset_px"] == pytest.approx(5.9, abs=0.3)
    assert s74["centroid_offset_px"] < 1.0           # the windowed centroid alone would have passed
    assert s74["verdict"] == "off_target"


@pytest.mark.live
def test_live_pi_men_saturated_uses_photocenter():
    r = cc.check(261136679, 6.2678139196372324, 3901.578202, 2.789, sectors=[1, 4, 8, 11, 12, 13], plot=False)
    assert r["verdict"] == "on_target" and "photocenter" in r["verdict_basis"]
