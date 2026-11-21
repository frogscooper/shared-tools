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
    with pytest.raises(ValueError):
        cp.load_signal(p, 2)


@pytest.mark.live
def test_live_pi_men_c_packet_on_target(tmp_path):
    sig = {"kind": "periodic", "period_d": 6.2678139196372324, "epoch_btjd": 3901.578202, "duration_h": 2.789,
           "depth_ppm": None, "rank": 1}
    s = cp.build(261136679, sig, tpf_sectors=[1, 4, 8, 11, 12, 13], out_root=tmp_path)
    assert s["results"]["centroid_check"]["verdict"] == "on_target"
    assert s["results"]["allsector_gate"]["verdict"] == "pass"
    assert s["status"] == "known_object"
    assert any(f.startswith("centroid_s") for f in s["files"])
