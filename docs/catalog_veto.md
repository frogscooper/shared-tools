# catalog-veto

Cross-matches a TIC (plus an optional trial period, epoch, and depth) against:

| source key | what | how |
|---|---|---|
| `exofop_toi`, `exofop_ctoi` | ExoFOP TOI / CTOI lists | CSV dump cached as `catalogs/toi_YYYY-MM-DD.csv` (ET date) |
| `nasa_pscomppars` | NASA Exoplanet Archive confirmed planets | TAP, `tic_id = 'TIC n'` |
| `prsa_tess_eb` | Prsa+2022 TESS EBs (J/ApJS/258/16) | CfA VizieR mirror (CDS 404s), cone of `--eb-radius` (21") |
| `gaia_dr3_eb` | Gaia DR3 `vari_eclipsing_binary` | Gaia TAP, cone of `--eb-radius` (21") |
| `gaia_dr3_neighbors` | Gaia DR3 neighbor table: separation, G, delta_G, RUWE | Gaia TAP, cone of `--neighbor-radius` (120" = 2'), kept separate from the EB radius |

```bash
catalog-veto 261136679
catalog-veto 288183107 --period 12.96393 --epoch 3319.20707 --depth 2140 --full --out veto.json
```

**Fail closed.** Every source reports `status` as `ok`, `empty`, or `down`, plus an `error` and a timestamp.
`summary.verdict` is one of:

- `known_object`: some flag is set (known_planet, toi, ctoi, eb, nearby_eb, period_matches_catalog).
- `incomplete`: no flags, but at least one source was down. A failed query is never read as "not in catalog".
- `clean`: every source answered and nothing matched.

When a source is down, its `last_good` result is shown with its fetch date. It never stands in for a live answer.
Exit codes: 0 clean, 1 known_object, 3 incomplete.

Notes:
- `known_planet` is set by a pscomppars match or by a TOI disposition of KP or CP.
- `period_aliases` lists catalog periods that equal the trial period or relate to it by n or 1/n (n ≤ 5,
  1% tolerance), with the epoch offset in hours.
- Gaia positions are moved back to epoch 2000 with proper motion before they are compared with the TIC
  position, so stars with high proper motion such as pi Men still match. The target is the TIC's Gaia
  source id, or failing that the brightest source within 2".
- With `--depth`, each neighbor gets `needed_eclipse_depth`, the eclipse it would need to fake the depth
  if fully blended, and `could_mimic_depth`. A proper dilution and centroid tool comes in a later round.
- WASP-62 shows up in Prsa+2022 at twice the planet period. That gives a known planet with `eb: true`, and a
  note explains it.
