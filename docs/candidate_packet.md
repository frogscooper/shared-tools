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

It never says "planet".

## Single dips and duos

```bash
candidate-packet 230064179 --single-epoch 3515.8593 --duration 5.38 --depth 1996 --tpf-sectors 81
candidate-packet 232608943 --t1 1785.96 --t2 2751.5986 --duration 10.85 --depth 12000 --tpf-sectors 17,53
```

Folder `TIC<id>_single_<epoch>/` or `TIC<id>_duo_<t1>_<t2>/`:

- `events.png`: each event zoomed, with its trapezoid fit (red) and V fit (blue dashed).
- `full_lc.png`: all sectors, with the events marked.
- `aliases.png` and an alias table in packet.md (duo only, from `duo-solver`). The table lists the surviving
  periods, how many predicted transits each has in coverage, the expected b=0 T14, and the next predicted window in UTC.
- `centroid_e<i>_s<N>.png`: a difference image per event. With a single event per TPF, the out-of-transit
  image uses only cadences within max(1 d, 4·T14) of the event. A whole-sector baseline let drifts win: on
  TOI-4600 c the brightest difference pixel was 4.5 px off with it and 0.03 px off with the local baseline.
- Dilution table from the Gaia neighbors, and catalog_veto.

Status:
- `duo_candidate_needs_human_review`: the duo gate passed, the label is still "duo candidate", every event's centroid is
  on target, and catalog_veto is clean.
- `single_needs_human_review`: the dip is flat-bottomed, on target, and catalog-clean.
- Otherwise `unpromoted` or `known_object`. A dip that `cross_star_veto` flagged as common-mode stays `unpromoted`.

The candidates.csv row has `kind` single or duo, `t2_btjd`, and `surviving_periods`. Rows are keyed by TIC, kind and first epoch.
