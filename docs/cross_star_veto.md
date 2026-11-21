# cross-star-veto and cadence masks

```bash
cross-star-veto --dips tests/fixtures/hunter_overnight_dips.csv --tol 1.0 --max-stars 1
cross-star-veto --tics 1,2,3 --sector 74 --out-dir veto_s74         # fetch, mask, search, veto
cross-star-veto --dips dips.csv --events candidates.csv --tol 0.5 --max-stars 2
```

## Masks (`shared_tools.exo.masks.cadence_mask`)

`good, why = cadence_mask(time, quality, flux, author)` returns a boolean mask, plus the dropped
cadences per reason so you can count them:

| reason | rule |
|---|---|
| quality | lightkurve "default" bad bits (17087). For QLP, any nonzero flag. |
| momentum_dump | Desat bit (32), padded ±0.05 d |
| scattered | Straylight / Straylight2 bits (2048, 4096) |
| orbit_edge | first and last `edge_d` (1 d) of every stretch split at gaps > 0.5 d |
| nonfinite | NaN time or flux |

`dips_from_lightcurves` applies the mask, runs a biweight detrend and the `transit_search` single-dip box, and returns a
dip table. Orbit edges are handled by `edge_policy`:
- `"flag"` (default): edge zones are searched, but a dip centred there needs SNR ≥ 10 and is marked `in_orbit_edge`.
- `"drop"`: edge zones are removed before the search.

`"flag"` is the default because real transits do sit there: TOI-4600 c's sector 17 transit ends 0.4 d before the
orbit does, and with `"drop"` the control is lost.

## Veto

For each dip, `n_stars_at_epoch` is 1 (itself) plus the number of distinct other stars with a dip within `tol_d`.
`common_mode` is set when that count is above `max_stars`, or above `max_frac` × the number of stars searched.

The CCD-level table `common_mode_epochs.csv` clusters all dip times by single linkage at `tol_d`. It keeps clusters seen
on more than `max_stars` stars and lists `t_mid`, the span, `n_stars`, `frac_stars` and the TICs.

Output files: `dips_flagged.csv`, `common_mode_epochs.csv`, `mask_summary.csv` (light-curve mode) and `summary.json`.

## Regression: hunter's 21 Nov overnight run

`tests/fixtures/hunter_overnight_dips.csv` holds all 603 SNR ≥ 8 dips (65 stars) from the 200-star batch, taken from
`/workspace/exoplanet/out/results/*.json`. `hunter_overnight_flags.csv` holds the 32 raw flags with hunter's post-vet verdict.

With hunter's rule (1 d tolerance, flag when at least one other star dips, so `max_stars=1`), 24 of the 29 single-transit
flags are vetoed. That matches hunter's 24 of 32 row for row, and includes all three near-misses. The other 3 of the 32
were periodic flags, vetoed for other reasons.

The CCD table at `tol=0.5, max_stars=4` puts the worst epochs in sector 86 (BTJD 3644.2 and 3657.8, 13 stars each) and
sector 74 (3335.2, 12 stars).
