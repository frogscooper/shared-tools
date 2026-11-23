# bls-triage

```bash
bls-triage --run-dir runs/s74_c1_d2                    # reads <run>/stages/bls/*.json written by hunt
bls-triage --run-dir runs/s74_c1_d2 --snr-min 8 --all-signals
```

These are the cuts from the 21 Nov S74/1/2 hunt, written as code, plus two checks that hunt did by hand.
Only the top periodic signal of each star is triaged unless you pass `--all-signals`.

| cut | fails when | soft? |
|---|---|---|
| snr | BLS SNR < 9 | yes (near miss if ≥ 7) |
| n_transits | < 3 transits with data | yes |
| odd_even | odd/even ≥ 3σ | |
| dominance | top SNR < 1.5× the next-best periodic signal | yes |
| harmonic | within 5% of 6.85 / 13.7 / 27.4 d | |
| integer_day | \|P − round(P)\| < 0.03 d (P ≥ 0.9 d) | |
| radius | sqrt(depth) · R* ≥ 20 Re | |
| grid_edge | duration at the top of the band's grid and P < 3 d | |
| duration | [duration-check](duration_check.md) fails for the TIC density (grazing or too long) | |
| systematic | ≥ 1/3 of in-coverage transits fall in a CCD-wide bad window | yes |

**Bad windows**: 0.5-day bins where many stars have a single dip at once. The threshold is at least
5 stars and at least 5× the median count per bin. This automates the BTJD 2744–2751 (S53) cluster seen
by hand on 21 Nov.

A signal that fails exactly one soft cut is a **near miss**. Survivors and near misses go to the gate.
Outputs are `triage.csv` (every star's top signal, with its fails) and `triage.json` (counts, windows,
survivors, near misses).
