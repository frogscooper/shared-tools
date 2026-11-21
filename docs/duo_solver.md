# duo-solver

```bash
duo-solver 232608943 --t1 1785.96 --t2 2751.5986 --duration 10.85 --depth 12000 --pmin 10 --rho 1.45
duo-solver 209464063 --t1 1424.711 --t2 2140.665 --duration 6.86 --depth 2400 --mstar 0.905 --rstar 0.938 --sectors 4,30,31
```

1. **Events.** Each event is refit with a least-squares trapezoid (t0, depth, T14, ingress fraction), averaged over the
   exposure for 30-min data.
   - Errors on t0 and T14 come from a Δχ² profile, not the Jacobian.
   - If an ingress or egress falls in a data gap (`partial`), the gap width / √12 is added to the T14 error, and half of
     that to the t0 error. TOI-4600 c's sector 17 ingress is in a 3 h gap.
2. **Aliases.** P_n = (T2 − T1)/n for n = 1 … ⌊ΔT/pmin⌋. Each predicted transit inside good data (any of SPOC or
   TESS-SPOC, all sectors) is measured with `allsector_gate.measure_event`: box depth on the detrended flux, with a local
   baseline.
   - The prediction has a timing error, σ_k² = ((1 − k/n)σ1)² + ((k/n)σ2)². Every centre within ±3σ_k is tried, and the
     most dip-like one is kept.
   - Window verdicts:
     - `excluded`: (ref − depth)/err ≥ 3 and depth/err < 3, using the same box estimator measured at the two events.
     - `seen`: depth/err ≥ 3 and within 3σ of ref, with a 10% depth floor.
     - Anything else is `unresolved`.
   - Alias status: `excluded` beats `seen`, which beats `unresolved`, which beats `open` (no window in data).
3. **Duo gate**, hard pass/fail:
   - (a) `events_consistent`:
     - Depths agree within 3σ (5% floor) and T14 agrees within 3σ.
     - Each event is flat-bottomed: the trapezoid beats a V by Δχ² ≥ 9, and ingress/T14 < 0.4.
     - A grazing planet fails this, by design.
   - (b) `coverage_excludes`: no alias is left `unresolved`. It fails closed.
   - (c) `label`: "duo candidate" until a predicted transit is `seen`. After that, the label names the alias that explains
     the most windows. Windows at T2 + m·ΔT are shared by every alias, so ties go to the longest period.
4. **Per alias:** the next 3 predicted windows (UTC and ET, ±(3σ + T14/2)), and, with `--rho` or `--mstar`/`--rstar`, the
   T14 expected at b = 0 (`stellar_density.expected_duration_h`).

Outputs: `duo.json` and `aliases.csv` under `<cache>/duo_solver/TIC<id>_<t1>_<t2>/`.

## Regressions (live)

- **TOI-4600 c** (Mireles et al. 2023, P = 482.8191 d). Transits in S17 (TESS-SPOC 30 min) and S53 (SPOC 2 min), with
  38 sectors of coverage.
  - 96 aliases with P ≥ 10 d; 94 excluded by measured windows. P = 321.87 d is excluded by S41 (−2 ± 192 ppm where
    12 000 ppm was predicted).
  - Survivors: 965.612 and 482.806 ± 0.009 d (the published value is within 1.4σ). Duo gate: pass.
- **TOI-5678 b** (Ulmer-Moll et al. 2023, P = 47.73 d), S4 + S30.
  - With S4, S30 and S31 only: 71 aliases, 47 excluded, 47.730 among the 24 survivors.
  - With the later sectors (S97, S106, S107): the 47.730 alias has 3 predicted transits `seen`. The label switches to
    "no longer a duo" and names 47.730 d.
