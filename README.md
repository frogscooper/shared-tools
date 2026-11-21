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

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e . --no-deps

tess-lc-fetch 261136679 --author SPOC --exptime 120
catalog-veto 149603524 --period 4.4119 --epoch 3982.0655
allsector-gate 288183107 --period 12.96393 --epoch 3319.20707 --duration 6.208 --depth 2140 --claimed-sectors 74,75
period-harmonic-flag 13.622518
```

Cache: `$SHARED_TOOLS_CACHE` (default `~/.cache/shared_tools`). Re-running any tool reuses
the cache, so a second `tess-lc-fetch` downloads nothing and returns the same `manifest_hash`.
Pass `--refresh` to force a live re-query.

Time convention: BTJD = BJD_TDB - 2457000 everywhere. Catalog epochs in full BJD are converted.

## Tests

```bash
pytest -q          # offline: mocked archives + WASP-62 b control on a 64 KB committed fixture
pytest -q --live   # also hits MAST / ExoFOP / NASA / Gaia / VizieR (pi Men, WASP-62, TIC 288183107, 154566531, 105755855)
```

CI (`.github/workflows/tests.yml`) runs the offline suite and a WASP-62 b CLI smoke test on every push.
`scripts/make_wasp62_fixture.py` rebuilds the fixture from the SPOC sector 1 and 2 files.
