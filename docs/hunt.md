# hunt

One resumable command per CCD (or TIC list): controls, selection, parallel downloads, the duo funnel,
the BLS search, triage, the all-sector gate and vetting. It ends with a short summary and a hunt-log draft.

```bash
hunt run --sector 74 --camera 1 --ccd 2 --run-dir runs/s74_c1_d2           # 150 s chunks; rerun to continue
hunt run --sector 74 --camera 1 --ccd 2 --run-dir runs/s74_c1_d2 --detach  # whole run in the background
hunt status --run-dir runs/s74_c1_d2                                       # ~10 lines
hunt summary --run-dir runs/s74_c1_d2                                      # prints summary.md
```

## Why

The 21 Nov S74/1/2 hunt was done by hand. It meant chunking the work around the 3-minute shell limit,
running thread and process pools written on the spot, reading big JSON outputs, applying the triage
cuts in ad-hoc code, gating survivors one at a time, and fitting the best signal to find that it was
grazing. `hunt` does all of that, and you only have to read `summary.md`.

## Time budget

`--budget N` (default 150 s): no new work starts after N seconds. Items already running finish, progress
is saved, and the exit code is **75**. Run the same command again to continue. Every item is a cached
stage file, so nothing is ever recomputed. `--budget 0` has no limit. `--detach` reruns itself with
`--budget 0` in a new session, logs to `<run>/hunt.log`, and returns at once.

## Stages

| stage | what | parallel |
|---|---|---|
| controls | TOI-4600 c through the duo funnel; pi Men c through BLS (P within 0.002 d). Cached under `$SHARED_TOOLS_CACHE/hunt_controls/<code hash>/`, so they rerun only when the tool code changes. A failed control stops the hunt (exit 2). | |
| select | `batch_runner` select: TESS-SPOC targets on the CCD, TIC cuts, known TOI/CTOI removed | |
| fetch | coverage + light curves for every selected star | `--dl-threads` (4) |
| dips | single-dip search (stars with early and late coverage) | `--workers` |
| duo | `batch_runner` in one pass, with the in-sector rule dropped and each pair labelled `in_sector`. It merges dip fragments, skips EB-sized pairs, and demotes stars already shown periodic. | |
| bls | `transit_search`: pmin 0.5, pmax 60, 2 signals | `--workers` |
| triage | `bls-triage` defaults (see [bls_triage.md](bls_triage.md)) | |
| gate | `allsector_gate` with the TIC density (duration_density test) on survivors and near misses (`--max-gate` 40) | `--workers` |
| vet | `candidate_packet` for gate passes: catalog (incl. Gaia NSS/RUWE), secondary and dilution, centroid on the last `--tpf-sectors` (4) sectors | |
| summary | `summary.md`, `hunt-log-draft.md` | |

A duo pair whose two dips both line up with a BLS periodic signal (SNR ≥ 7) of the same star is labelled
"explained by BLS periodic" in the summary and isn't counted as a duo candidate.

## Outputs (`<run>/`)

- `summary.md`: under 80 lines. It has the headline, controls, funnel, duo table, triage cut counts, shared-epoch windows, gate table, vet table and errors.
- `hunt-log-draft.md`: the project's hunt-log layout with the numbers filled in. Edit the `<!-- edit -->` spots and save it to the project.
- `triage.csv` / `triage.json`, `funnel.csv` / `funnel.json`, `duo_pairs.csv`, `gate/`, `packets/`, `bls/`.
- `progress.json`: per-stage counts. `hunt status` reads it.

## Token-saving habits

- Read `summary.md`, not the stage JSON.
- Use `--brief` on `allsector-gate` and `transit-search` for one-line answers.
- Use `bls-triage --run-dir` to re-cut with other thresholds. It takes seconds and recomputes nothing.
- For a background run, poll with `hunt status` (about 10 lines), not `cat hunt.log`.

Speed: the banded BLS grid runs about 1.8× faster per star than 0.3.0 (synthetic 4-sector benchmark, 5/5
injections recovered at SNR ≈ 15). With 2 CPUs a 2,000-star CCD needs about 2–3 h of BLS.
