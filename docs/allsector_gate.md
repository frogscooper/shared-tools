# allsector-gate

Checks one trial ephemeris against every available sector. It uses `tess_lc_fetch --pick best`,
or the `--lc-file` FITS/CSV files you pass.

```bash
allsector-gate 261136679 --period 6.2678139 --epoch 3901.578202 --duration 2.789
allsector-gate 154566531 --period 13.622518 --epoch 3317.83311 --duration 4.19 --depth 3049 --claimed-sectors 74,75
```

Method: each sector gets a wotan biweight fit (window = max(0.5 d, 3 × duration)) with the predicted
transits masked. Each predicted epoch then gets:
`depth_ppm ± depth_err_ppm` (central 80% of the transit vs 1 to 3 durations either side), point counts
(`n_in`, `n_in_expected`, `n_oot`, `n_pre`, `n_post`), the quality bits and their names, and flags:
`missing_coverage`, `quality_bits_in_window` (>10% of the window flagged by the lightkurve default mask),
`stray_light`, `momentum_dump`, `near_edge` (within one duration of a sector edge or of a gap longer than 0.5 d),
and `one_sided_baseline`. Status is `clean`, `flagged`, or `no_coverage`. Only clean events feed the tests.

| test | fails when | incomplete when |
|---|---|---|
| orbit_harmonic | period within 5% of 6.85 / 13.7 / 27.4 d | never |
| missing_transit | a clean event is ≥3σ below the reference depth and <3σ above zero | <2 clean events |
| repeats | not detected (≥3σ) in ≥2 sectors, or, with `--claimed-sectors`, in ≥1 sector outside them, even though such sectors have clean coverage | no independent clean coverage |
| odd_even | \|odd − even\| / σ > 3 (hard gate) | either parity has no clean event (fails closed) |
| depth_consistency | one detected event >5σ from the weighted mean of the others (10% of the depth added as a systematic floor) | <3 detected events |
| secondary | phase-0.5 dip with SNR ≥5 and ≥10% of the primary depth | too few points at phase 0.5 |

Reference depth: `--depth` if given, else the mean of clean events in `--claimed-sectors`, else the mean of all clean events.
For odd/even and depth consistency, per-event errors are scaled by a robust excess-scatter factor
(1.4826 × MAD of normalized residuals, floored at 1), because white-noise errors understate red noise.

Verdict: `fail` if any test fails; otherwise `incomplete` if any test couldn't run; otherwise `pass`.
Output: `gate.json` and `gate.png` (per-event depths, odd/even fold, phase-0.5 fold) under
`<cache>/allsector_gate/TIC<id>_P<period>/` or `--out-dir`. Exit codes: 0 pass, 1 fail, 3 incomplete.

Out of scope for now: difference images, centroids, and dilution. Those come in a later round.
