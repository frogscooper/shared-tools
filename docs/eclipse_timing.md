# eclipse-timing

```bash
eclipse-timing 141618383 --period 10.43181 --epoch 2758.99727 --duration 1.0 --depth 1386
```

Do the transits or eclipses arrive on time? It fits one trapezoid to the folded light curve. Then for each
event only the mid-time moves: a χ² scan over ±30% of T14, with errors from Δχ² = 1 (inflated when the
event is noisier than white noise). Events without both ingress and egress are skipped. It fits a weighted
linear ephemeris and reports O−C and χ²_red.

The `timing_variation` flag needs ≥ 6 events, χ²_red ≥ 4, and max |O−C| ≥ 3× the median error.
Synthetic check (4 sectors, 28 transits, 1.5 min errors):
- no TTV: χ²_red 1.11, not flagged
- 3 min TTV: χ²_red 4.4, flagged
- 20 min TTV: χ²_red 47, flagged

TTVs on a planet suggest another planet; ETVs on an EB suggest a third body. Spots, detrending and blends
also shift times, so a flag is a lead for follow-up.
