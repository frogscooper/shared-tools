# duration-check

```bash
duration-check 10.4317 --t14 1.2 --depth 1370 --rho 0.78      # TIC 141618383: fail, too_short (b ≈ 1.00)
duration-check 6.26782 --t14 2.95 --depth 290 --rho 0.83      # pi Men c: pass
```

Is a transit duration possible for this star, period and depth on a circular orbit? It takes
k = sqrt(depth) and a/Rs from ρ* and P, then solves T14 = P/π · asin(sqrt((1+k)² − b²)/(a/Rs)) for b.
It also reports the ρ* a central transit would need.

- **too_short**: needs b > 1 − k, which is grazing. A grazing body this deep is usually an eclipsing binary, and k is then only a lower limit.
- **too_long**: T14 > 1.5× the b = 0 maximum. The host is less dense than the TIC says, or it's a blend, or the orbit is eccentric.

It's used by `allsector-gate --rho` (as the `duration_density` test), by `bls-triage`, and by `hunt`. On 21 Nov
this needed a batman + emcee fit by hand for TIC 141618383; this check flags it in microseconds. BLS
durations are coarse box widths, so the limits are loose on purpose.
