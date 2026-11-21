# period-harmonic-flag

```bash
period-harmonic-flag 13.622518 26.525555 27.142599 16.1329819 --tol 0.05
```

```python
from shared_tools.exo.harmonics import period_harmonic_flag
period_harmonic_flag(13.622518)   # {"flagged": True, "near": [13.7], "refs": [{ref_d, offset_d, offset_pct, flagged}, ...]}
```

Reports the offset in days and in percent from 6.85, 13.7, and 27.4 d (half, one, and two TESS orbits).
`flagged` is true when any |offset| / ref ≤ tol (default 5%). `allsector-gate` uses this as a hard fail.
