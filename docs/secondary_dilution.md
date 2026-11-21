# secondary-dilution

```bash
secondary-dilution 149603524 --period 4.411937 --epoch 3982.065462 --duration 3.632
secondary-dilution 288183107 --period 12.96393 --epoch 3319.20707 --duration 6.208 --depth 2140 --veto-json veto.json
```

- **Secondary:** box depth (width 0.8 × duration) at phase 0.5, plus a scan of every other phase in
  steps of a quarter duration.
  - Depths are measured per orbit, combined after a 5σ clip, and given the larger of the white-noise
    and orbit-to-orbit errors (`folding.phase_depth`). This stops a single scattered-light ramp from
    faking a 25σ "secondary" (it did on pi Men sector 4 before the clip).
  - `secondary_at_0p5`: SNR ≥ 3.
  - `dip_elsewhere_in_phase`: SNR ≥ max(5, expected noise maximum + 2).
- **Dilution:** for each Gaia DR3 neighbor from `catalog_veto` (or from `--veto-json`):
  - `flux_fraction = 1/(1+10^(0.4 ΔG))`
  - `needed_eclipse_depth = depth / flux_fraction`
  - `could_cause` = needed ≤ 1 and separation ≤ 63" (3 px). This assumes all of the neighbor's light
    falls in the aperture, so it is conservative.
