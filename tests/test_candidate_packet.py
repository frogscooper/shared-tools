from pathlib import Path
import csv
import json

import pytest

from shared_tools.exo import candidate_packet as cp

WASP62 = {"kind": "periodic", "period_d": 4.411937227697335, "epoch_btjd": 3982.065462, "duration_h": 3.632,
          "depth_ppm": None, "snr": 400, "rank": 1, "status": "unpromoted"}


def test_offline_packet_is_complete_and_idempotent(fixtures, tmp_path):
    lc = [fixtures / "wasp62_s1_s2_10min.csv.gz"]
    s = cp.build(149603524, WASP62, lc_files=lc, out_root=tmp_path, offline=True, neighbors=[])
    files = set(s["files"])
    assert {"packet.md", "signal.json", "full_lc.png", "fold.png", "odd_even.png", "events.png", "gate.png",
            "secondary.png"} <= files
    assert s["results"]["allsector_gate"]["verdict"] == "pass"
    assert s["status"] == "unpromoted"            # no centroid, no catalog -> not promoted
    assert "Not a detection" in (tmp_path / "TIC149603524_P4.41194" / "packet.md").read_text()
    cp.build(149603524, WASP62, lc_files=lc, out_root=tmp_path, offline=True, neighbors=[])
    rows = list(csv.DictReader((tmp_path / "candidates.csv").open()))
    assert len(rows) == 1 and rows[0]["gate_verdict"] == "pass"


def test_load_signal_from_search_json(tmp_path):
    p = tmp_path / "signals.json"
    p.write_text(json.dumps({"signals": [{**WASP62, "rank": 1}, {"kind": "single", "rank": 2, "epoch_btjd": 1.0,
                                                                  "duration_h": 4}]}))
    assert cp.load_signal(p, 1)["period_d"] == WASP62["period_d"]
    assert cp.load_signal(p, 2)["kind"] == "single"
    with pytest.raises(ValueError):
        cp.load_signal(None, period=None, epoch=1.0, duration=3.0)


def _synth_duo():
    from synth import make_sector

    tr = (6 / 24, 0.1)
    return [make_sector(1, 1000.0, [(1010.0, 3000e-6, *tr)]), make_sector(2, 1027.5, []),
            make_sector(3, 1300.0, []), make_sector(4, 1700.0, [(1710.0, 3000e-6, *tr)])]


def test_duo_packet_offline(tmp_path):
    sig = cp.load_signal(t1=1010.0, t2=1710.0, duration=6.0, depth=3000)
    s = cp.build(1, sig, out_root=tmp_path, offline=True, secs=_synth_duo())
    d = Path(s["packet_dir"])
    assert {"packet.md", "signal.json", "full_lc.png", "events.png", "aliases.png"} <= set(s["files"])
    md = (d / "packet.md").read_text()
    assert "Duo gate" in md and "duo candidate" in md and "| 10 | 70.0" in md
    assert s["status"] == "unpromoted"          # no centroid, no catalog: never promoted offline
    rows = list(csv.DictReader((tmp_path / "candidates.csv").open()))
    assert rows[0]["kind"] == "duo" and "70.0" in rows[0]["surviving_periods"]
    cp.build(1, sig, out_root=tmp_path, offline=True, secs=_synth_duo())       # idempotent row
    assert len(list(csv.DictReader((tmp_path / "candidates.csv").open()))) == 1


def test_single_packet_offline(tmp_path):
    from synth import make_sector

    secs = [make_sector(7, 2000.0, [(2008.0, 2500e-6, 5 / 24, 0.1)], cad_min=10.0)]
    s = cp.build(2, cp.load_signal(single_epoch=2008.02, duration=5.0, depth=2500), out_root=tmp_path, offline=True,
                 secs=secs)
    assert s["results"]["event_shape"]["verdict"] == "pass"
    assert abs(s["events"][0]["t0_btjd"] - 2008.0) < 0.01
    assert {"packet.md", "events.png", "full_lc.png"} <= set(s["files"]) and s["status"] == "unpromoted"


@pytest.mark.live
def test_live_toi4600c_duo_packet(tmp_path):
    sig = cp.load_signal(t1=1785.96, t2=2751.5986, duration=10.85, depth=12000)
    s = cp.build(232608943, sig, tpf_sectors=[17, 53], out_root=tmp_path)
    assert s["results"]["catalog_veto"]["verdict"] == "known_object"
    assert any(abs(a["period_d"] - 482.82) < 0.05 for a in s["results"]["duo_solver"]["surviving"])
    assert len(s["results"]["centroid_check"]["per_event"]) >= 1


@pytest.mark.live
def test_live_pi_men_c_packet_on_target(tmp_path):
    sig = {"kind": "periodic", "period_d": 6.2678139196372324, "epoch_btjd": 3901.578202, "duration_h": 2.789,
           "depth_ppm": None, "rank": 1}
    s = cp.build(261136679, sig, tpf_sectors=[1, 4, 8, 11, 12, 13], out_root=tmp_path)
    assert s["results"]["centroid_check"]["verdict"] == "on_target"
    assert s["results"]["allsector_gate"]["verdict"] == "pass"
    assert s["status"] == "known_object"
    assert any(f.startswith("centroid_s") for f in s["files"])
