import pytest

from shared_tools.exo.harmonics import period_harmonic_flag


@pytest.mark.parametrize("p,near", [(13.622518, 13.7), (26.525555, 27.4), (27.142599, 27.4), (6.9, 6.85)])
def test_flags_near_tess_harmonics(p, near):
    r = period_harmonic_flag(p)
    assert r["flagged"] and near in r["near"]


@pytest.mark.parametrize("p", [16.1329819, 12.963929714582635, 4.4119, 6.2678])
def test_does_not_flag(p):
    assert not period_harmonic_flag(p)["flagged"]


def test_reports_offset_and_percent():
    ref = {r["ref_d"]: r for r in period_harmonic_flag(13.622518)["refs"]}
    assert set(ref) == {6.85, 13.7, 27.4}
    assert ref[13.7]["offset_d"] == pytest.approx(-0.077482, abs=1e-6)
    assert ref[13.7]["offset_pct"] == pytest.approx(-0.566, abs=1e-3)
