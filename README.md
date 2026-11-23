# shared-tools

benji's shared toolbox for his agents. Small, plain Python with no frameworks. Every tool
here reads public archives only. Nothing submits, posts, or contacts anyone.

## Exoplanet tools (`shared_tools/exo/`)

Built from the exoplanet hunter and exoplanet vetter wish lists. A "pass" from any of these
means "worth a human look". It never means "planet", and the known planets used in the tests
are controls only.

| tool | what it answers | doc |
|---|---|---|
| `tess-lc-fetch` | Which public SPOC / TESS-SPOC / QLP light curves exist for this TIC? Download them once, cache them, and write a manifest. | [docs/tess_lc_fetch.md](docs/tess_lc_fetch.md) |
| `catalog-veto` | Is this TIC already a known planet, TOI, CTOI, or eclipsing binary? Who are its Gaia neighbors? Fails closed when a catalog is down. | [docs/catalog_veto.md](docs/catalog_veto.md) |
| `allsector-gate` | Does a trial ephemeris hold up in every sector? Checks missing transits, repeats, odd/even, depth consistency, secondary eclipse, and orbit harmonics. | [docs/allsector_gate.md](docs/allsector_gate.md) |
| `period-harmonic-flag` | Is this period within 5% of the 6.85 / 13.7 / 27.4 d TESS orbit harmonics? | [docs/period_harmonic_flag.md](docs/period_harmonic_flag.md) |
| `transit-search` | Detrend, then run a BLS periodic search and a single-transit dip finder. Every signal comes out "unpromoted". | [docs/transit_search.md](docs/transit_search.md) |
| `centroid-check` | Is the dip on the target? Difference-image centroid, brightest difference pixel, depth vs aperture, photocenter shift. | [docs/centroid_check.md](docs/centroid_check.md) |
| `secondary-dilution` | Is there a phase-0.5 or eccentric secondary? Which Gaia neighbors are bright enough to fake the depth? | [docs/secondary_dilution.md](docs/secondary_dilution.md) |
| `candidate-packet` | One review folder: packet.md, signal.json, all plots, and a candidates.csv row. | [docs/candidate_packet.md](docs/candidate_packet.md) |
| `cross-star-veto` | Cadence masks (quality bits, momentum dumps, scattered light, orbit edges), then flag dip epochs shared by many stars on one CCD. | [docs/cross_star_veto.md](docs/cross_star_veto.md) |
| `duo-solver` | Two single transits: list the allowed periods P = ΔT/n, drop every alias the measured coverage rules out, and run the duo gate. | [docs/duo_solver.md](docs/duo_solver.md) |
| `batch-runner` | Resumable duo funnel over TICs or one sector/camera/CCD, with a funnel CSV and a known-duo control. | [docs/batch_runner.md](docs/batch_runner.md) |
| `stellar-density` | rho* = 3π/(G P²)(a/Rs)³ in cgs, compared with the TIC M/R³. | [docs/stellar_density.md](docs/stellar_density.md) |
| `duration-check` | Is this T14 possible for the star's density, period and depth? Flags grazing (likely EB) and too-long transits. | [docs/duration_check.md](docs/duration_check.md) |
| `bls-triage` | The CCD-wide BLS cuts (SNR, transits, odd/even, dominance, harmonics, integer days, radius, grid edge, duration, shared-epoch systematics) → survivors and near misses. | [docs/bls_triage.md](docs/bls_triage.md) |
| `eclipse-timing` | Do transits/eclipses arrive on time? Per-event mid-times, O−C, χ²; flags TTVs/ETVs (another planet, a third body). | [docs/eclipse_timing.md](docs/eclipse_timing.md) |
| `survey` | **Scan many sectors × cameras × CCDs fast**: a quick single-sector look at every star (direct URLs, files deleted after), the full `hunt` on flagged stars, then classes (planet, duo, single transit, EB) with catalog novelty and one leaderboard. | [docs/survey.md](docs/survey.md) |
| `hunt` | **One resumable command per CCD**: controls → select → parallel fetch → duo funnel → BLS → triage → gate → vet → `summary.md` + hunt-log draft. Time-boxed chunks or `--detach`. | [docs/hunt.md](docs/hunt.md) |

**Many CCDs:** `survey plan --dir surveys/s70-75 --sectors 70-75 && survey run --dir surveys/s70-75 --detach`,
then `survey top --dir surveys/s70-75`.

**Whole CCD:** `hunt run --sector 74 --camera 1 --ccd 2 --run-dir runs/s74_c1_d2` (rerun until exit 0, or add `--detach`),
then read `runs/s74_c1_d2/summary.md`. The steps below are what `hunt` runs for you.

Typical flow for periodic signals: `transit-search`, then `catalog-veto`, then `allsector-gate`, then `candidate-packet`.
Typical flow for long-period duos: `batch-runner` does select, coverage, light curves, masks and dips, `cross-star-veto`,
cross-year pairs, `duo-solver`, then a duo `candidate-packet`.
The packet runs the gate, centroid, secondary and catalog checks itself.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e . --no-deps

tess-lc-fetch 261136679 --author SPOC --exptime 120
catalog-veto 149603524 --period 4.4119 --epoch 3982.0655
allsector-gate 288183107 --period 12.96393 --epoch 3319.20707 --duration 6.208 --depth 2140 --claimed-sectors 74,75
period-harmonic-flag 13.622518
transit-search 261136679 --sectors 1,4,8,11,12,13 --pmax 30
centroid-check 288183107 --period 12.96393 --epoch 3319.20707 --duration 6.208 --sectors 74
secondary-dilution 149603524 --period 4.411937 --epoch 3982.065462 --duration 3.632
candidate-packet 261136679 --signals ~/.cache/shared_tools/transit_search/TIC261136679/signals.json --rank 1 --tpf-sectors 1,4,8,11,12,13
stellar-density 4.411937 --a-rs 9.8785 --mstar 1.28 --rstar 1.21
cross-star-veto --dips tests/fixtures/hunter_overnight_dips.csv --tol 1.0 --max-stars 1
duo-solver 232608943 --t1 1785.96 --t2 2751.5986 --duration 10.85 --depth 12000 --pmin 10 --rho 1.45
candidate-packet 232608943 --t1 1785.96 --t2 2751.5986 --duration 10.85 --depth 12000 --tpf-sectors 17,53
batch-runner --control toi4600c --run-dir runs/control_toi4600c
duration-check 10.4317 --t14 1.2 --depth 1370 --rho 0.78
hunt run --sector 74 --camera 1 --ccd 2 --run-dir runs/s74_c1_d2 --detach && hunt status --run-dir runs/s74_c1_d2
```

What changed in 0.4.0 (from the 21 Nov S74/1/2 hunt's tool-issue list):
1. Duo: dips within 1× duration of a stronger dip are merged before pairing (wing fragments of one eclipse no longer pair). Pairs implying Rp > 25 Re are skipped.
2. Duo: one pair labelled periodic demotes all of that star's pairs, and no packets are built for them.
3. `catalog-veto`: Gaia DR3 non-single-star solutions (all I/357 tables, one VizieR call) and target RUWE. SB2 / EB models veto (`nss_binary`). Other orbits and RUWE > 1.4 add a note. Gaia cones fall back to the VizieR mirror when the Gaia archive is down (DR4 downtime from 25 Nov 2026).
4. `duration-check`, plus the `allsector-gate --rho` `duration_density` test.
5. `transit-search`: per-period-band duration grids (up to 12 h at long P, no longer capped at 4 h), coarser coarse-search phase bins, period-grid oversample 2. That's about 1.8× faster with better long-duration SNR. Signals carry `duration_at_grid_edge`. `--brief`.
6. Speed: `hunt` parallelizes downloads (threads, MAST 429/503 backoff in `tess_lc_fetch`) and CPU stages (processes), and works within a time budget. `allsector-gate --brief`.

Cache: `$SHARED_TOOLS_CACHE` (default `~/.cache/shared_tools`). Re-running any tool reuses
the cache, so a second `tess-lc-fetch` downloads nothing and returns the same `manifest_hash`.
Pass `--refresh` to force a live re-query.

Time convention: BTJD = BJD_TDB - 2457000 everywhere. Catalog epochs in full BJD are converted.

## Tests

```bash
pytest -q          # offline: mocked archives + WASP-62 b control on a 64 KB committed fixture
pytest -q --live   # also hits MAST / ExoFOP / NASA / Gaia / VizieR (pi Men, WASP-62, TIC 288183107, 154566531, 105755855)
                   # and downloads TPFs (~50 MB each) for the centroid and packet tests
```

CI (`.github/workflows/tests.yml`) runs the offline suite and a WASP-62 b CLI smoke test on every push.
`scripts/make_wasp62_fixture.py` rebuilds the fixture from the SPOC sector 1 and 2 files.
