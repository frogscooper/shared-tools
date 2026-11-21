# tess-lc-fetch

Inventories every public SPOC, TESS-SPOC and QLP light curve for one or more TICs, keeps one
product per (author, sector, cadence), downloads anything that isn't cached yet, and returns a
`LightCurveCollection` plus a manifest.

```bash
tess-lc-fetch 261136679 --author SPOC --exptime 120   # pi Men, SPOC 2-min only
tess-lc-fetch 288183107 --pick best                    # one product per sector (SPOC > TESS-SPOC > QLP; 120 s over 20 s)
```

```python
from shared_tools.exo.tess_lc_fetch import fetch
lcc, manifest = fetch(261136679, authors=["SPOC"], exptimes=[120])
```

- Manifest: `<cache>/manifests/<key>.json` and `.csv`. Each row has tic, sector, author, cadence_s,
  product_uri, local_path, n_cadences, n_good (quality == 0), frac_good, crowdsap, flfrcsap, and the time range.
- `manifest_hash` is the sha256 of the product rows only (no timestamps), so it matches between cold and warm calls.
- The MAST inventory is cached for 24 h. If MAST is down, the tool falls back to the last inventory
  and says so (`inventory.source = "stale_cache"`).
- Downloads come from archive.stsci.edu first, because MAST's download API redirects to an S3 bucket
  that currently returns 403. A file only lands in the cache after it opens as FITS.
- A failed download is listed in `failures` and makes `complete: false`. Exit code 2.
- Duplicates: when two versions of the same product exist, the lexically last filename wins.
