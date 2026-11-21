# transit-search

```bash
transit-search 261136679 --sectors 1,4,8,11,12,13 --pmin 1 --pmax 30
```

1. One light curve per sector (via `tess_lc_fetch --pick best`, or `--lc-file`), the default quality
   mask, a wotan biweight detrend (`--window`, default 1 d), then 10-minute bins.
2. **Periodic:** astropy BoxLeastSquares (objective = SNR) on each observing season (sectors less than
   100 d apart). The log-period grid uses dP/P = shortest duration / (3 × season length).
   - Season peaks are clustered (0.5%), then refined phase-coherently on the full baseline (±0.5%).
   - Each signal's transits are masked before the next search (`--n-signals`).
   - A periodic peak is rejected (listed in `rejected_periodic`) if it has fewer than 2 transits with data,
     or if one event carries more than 60% of the signal.
3. **Single transits:** box widths 2 to 16 h, a local baseline on both sides, and an SNR normalized by the
   robust scatter of all trial boxes (red-noise aware). Default threshold 7.
   `matches_periodic` and `near_gap_or_edge` label dips without dropping them.

Output: `signals.json` and `signals.csv` with rank, kind, period_d / epoch_btjd, depth_ppm, duration_h,
SNR, SDE, n_transits, an odd/even preview (BLS odd vs even depth, in σ), `harmonic_flag`,
`tess_orbit_flag` (% offsets from 6.85, 13.7 and 27.4 d), and `harmonic_of`.
Every signal has `status: "unpromoted"` until `allsector-gate` and `catalog-veto` have run.

TLS is not used. BLS plus a phase-coherent refine recovers pi Men c at 6.26783 d (published 6.26782).
