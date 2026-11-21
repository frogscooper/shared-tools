# centroid-check

```bash
centroid-check 149603524 --period 4.411937 --epoch 3982.065462 --duration 3.632 --sectors 1,2
centroid-check 288183107 --period 12.96393 --epoch 3319.20707 --duration 6.208 --sectors 74
```

Pass the claimed-detection sectors in `--sectors`. TPFs come from `tess_lc_fetch` (`kind="tpf"`,
SPOC first, then TESS-SPOC) or from `--tpf-file`.

Per TPF:
- **Coordinates:** 0-indexed TPF-array pixels from `tpf.wcs.world_to_pixel`, after moving the TIC
  position (epoch 2000) to the TPF epoch with proper motion. `tpf.column` / `tpf.row` are never used,
  and `tests/test_centroid_check.py` fails if they are.
- **Difference image:** mean out-of-transit (more than 1.5 durations from mid-transit) minus mean
  in-transit (within 0.45 durations).
  - `centroid_offset_px`: flux-weighted centroid of the positive difference pixels within 4 px of the target. Flagged above 1 px.
  - `peak_offset_px`: brightest difference pixel. Flagged above 2 px. This catches the TIC 288183107 case,
    where the windowed centroid (0.4 px) looks fine but the peak sits 5.9 px away.
- **Depth vs aperture:** circular apertures of radius 1, 1.5, 2, 2.5, 3 and 4 px, plus the pipeline aperture,
  each with a local linear baseline per event. `depth_grows_with_aperture` is set when the 4 px depth is
  more than 1.5× the 1 px depth and the difference is above 3σ.
- **Photocenter shift:** the in- vs out-of-transit photocenter shift in the aperture gives the implied
  source position, `x_src = x_c - dx (1-δ)/δ`.
  - Saturated targets (brightest pixel > 8e4 e⁻/s, e.g. pi Men) bleed along columns, so for them
    this method decides the verdict.
  - Offsets are combined across sectors.
  - The verdict is `on_target` when the error is ≤ 1 px and the offset minus 3σ is ≤ 1 px.
- **Verdict:** `on_target`, `off_target`, or `inconclusive` (difference-image SNR below 3, or too few
  cadences). The PNG shows the out-of-transit image, the difference image with markers, and depth vs radius.
