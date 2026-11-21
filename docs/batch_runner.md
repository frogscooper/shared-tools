# batch-runner

```bash
batch-runner --control toi4600c --run-dir runs/control_toi4600c
batch-runner --sector 74 --camera 1 --ccd 2 --run-dir runs/s74_c1_d2 --max-stars 200
batch-runner --tics 1,2,3 --run-dir runs/mine --tmag-max 12.5 --redo duo
```

A resumable duo funnel. Each stage writes `<run>/stages/<stage>/<key>.json`, and a rerun reuses those files.
`--redo STAGE` recomputes one stage. An exception becomes that star's stage result (`ok: false`, plus the error) and does
not stop the run.

| stage | what | default cut |
|---|---|---|
| select | `--tics`, or the TESS-SPOC FFI target list for `--sector` mapped to `--camera`/`--ccd` with tess-point; TIC cuts; known-object prefilter (TIC in today's ExoFOP TOI/CTOI tables) | Tmag 8–13, Teff 3500–6000 K, logg ≥ 4.0 |
| coverage | SPOC/TESS-SPOC light curves in an early sector and a late one | ≤ 26 and ≥ 40 |
| lightcurves | one product per sector (`tess_lc_fetch`), min CROWDSAP | ≥ 0.9 |
| dips | `cadence_mask` (edge zones flagged, not dropped), biweight, single-dip box | SNR ≥ 7 (≥ 10 in an orbit edge) |
| veto | `cross_star_veto` per sector over all stars in the run; skipped below 10 stars | tol 0.5 d, > 2 stars |
| pairs | dips ≥ 200 d apart, depth and duration ratios within 0.5–2 (with `--sector`, one dip must be in it), top 20 by SNR | |
| duo | `duo_solver` with the TIC density | pmin 10 d |
| packet | duo `candidate_packet` for gate passes: per-event centroid, dilution, catalog_veto | |

Every cut is a flag (`--tmag-min`, `--teff-max`, `--crowdsap-min`, `--early-max-sector`, …).

Outputs:
- `funnel.csv`: the count surviving each stage.
- `funnel.json`: the counts, controls, every duo pair, and packets.
- `duo_pairs.csv`, `common_mode_epochs.csv`, `packets/`, and `duo/`.

`--control toi4600c` adds TOI-4600 c (TIC 232608943; P = 482.8191 d; transits at BTJD 1785.96 and 2751.60) and turns
off the known-object prefilter. The control passes if the pair is found and the true period survives `duo_solver`.

Control run on 21 Nov 2026: 12 dips, 20 pairs (most are TOI-4600 b transits, which `duo_solver` correctly labels
"periodic"). The S17+S53 pair passed the duo gate with survivors 965.61 and 482.81 d. Centroid on target, catalog
`known_object`. A rerun with everything cached takes 0.3 s.

Do not treat a run as a hunt result without a human review of each packet. The tool submits nothing.
