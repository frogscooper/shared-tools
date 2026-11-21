# candidate-packet

```bash
candidate-packet 261136679 --signals ~/.cache/shared_tools/transit_search/TIC261136679/signals.json --rank 1 --tpf-sectors 1,4,8,11,12,13
candidate-packet 149603524 --period 4.411937 --epoch 3982.065462 --duration 3.632 --lc-file tests/fixtures/wasp62_s1_s2_10min.csv.gz --offline
```

Writes `<cache>/packets/TIC<id>_P<period>/` (or `--out`):

- `packet.md`: signal table; gate verdict, tests and reasons; centroid table; secondary and dilution; catalog verdict.
- `signal.json`: the signal plus a summary of every sub-result, with paths to the full JSONs.
- `full_lc.png`, `fold.png`, `odd_even.png`, `events.png` (up to 12 event zooms), `gate.png`, `secondary.png`,
  and `centroid_s<N>.png` when TPFs were used.
- One row in `<packets>/candidates.csv`. A rerun replaces the row with the same TIC, kind and period, so the file stays idempotent.

Status:
- `known_object`: the catalog veto matched.
- `gate_passed_needs_human_review`: the gate passed, the centroid is on target, and there is no secondary.
- `unpromoted`: anything else, including when the centroid or catalog check didn't run.

It never says "planet". Single-transit packets are not supported yet.
