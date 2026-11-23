"""tess_lc_fetch: inventory, de-duplicate, download and cache public TESS light curves.

For each TIC it asks MAST for every SPOC, TESS-SPOC and QLP light curve, keeps one
product per (author, sector, cadence), downloads what is not already cached, and
writes a manifest (JSON + CSV). Re-running with the same inputs reuses the cache
and produces a manifest with the same `manifest_hash`.

    tess-lc-fetch 261136679 --author SPOC --exptime 120
    tess-lc-fetch 288183107 --pick best          # one product per sector

Python:
    from shared_tools.exo.tess_lc_fetch import fetch
    lcc, manifest = fetch(261136679, authors=["SPOC"], exptimes=[120])
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits
from astropy.table import Table

from .common import canonical_hash, default_cache_dir, jsonable, now_et, parse_tic, write_json

AUTHORS = ("SPOC", "TESS-SPOC", "QLP")
# Preference when --pick best: short-cadence SPOC first, then FFI products.
AUTHOR_RANK = {"SPOC": 0, "TESS-SPOC": 1, "QLP": 2}
MAST_DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file"
INVENTORY_COLS = ["author", "sector", "exptime", "productFilename", "dataURI", "size"]
MANIFEST_COLS = [
    "tic", "sector", "author", "cadence_s", "product_uri", "filename", "local_path",
    "n_cadences", "n_good", "frac_good", "n_finite_flux", "crowdsap", "flfrcsap", "t_start_btjd", "t_end_btjd",
]
UA = {"User-Agent": "shared-tools tess_lc_fetch (read-only research use)"}


# ---------------------------------------------------------------- inventory
THROTTLE_TRIES = 5
THROTTLE_BASE_S = 4.0


def _throttled(exc) -> float | None:
    """Seconds to wait if exc looks like MAST throttling or a transient outage (429/503/timeout), else None."""
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    if code in (429, 502, 503, 504):
        ra = (resp.headers or {}).get("Retry-After") if resp is not None else None
        try:
            return float(ra) if ra else THROTTLE_BASE_S
        except ValueError:
            return THROTTLE_BASE_S
    text = f"{type(exc).__name__}: {exc}"
    if any(k in text for k in ("429", "Too Many Requests", "503", "Service Unavailable", "Timeout", "timed out",
                               "ConnectionError", "RemoteDisconnected")):
        return THROTTLE_BASE_S
    return None


def with_backoff(fn, *args, tries=THROTTLE_TRIES, **kw):
    """Call fn, retrying with exponential backoff (+ jitter) on throttling. Parallel fetchers rely on this."""
    import random

    for i in range(tries):
        try:
            return fn(*args, **kw)
        except Exception as exc:
            wait = _throttled(exc)
            if wait is None or i + 1 == tries:
                raise
            time.sleep(wait * (2 ** i) * (0.75 + 0.5 * random.random()))


def search_mast(tic: int, authors, kind: str = "lc") -> list[dict]:
    """Live MAST query through lightkurve. kind = 'lc' or 'tpf'. Returns plain dict rows."""
    import lightkurve as lk

    fn = lk.search_lightcurve if kind == "lc" else lk.search_targetpixelfile
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr = with_backoff(fn, f"TIC {tic}", mission="TESS", author=list(authors))
    rows = []
    for r in sr.table:
        # Positional searches can return neighbors; keep only this TIC.
        if str(r["target_name"]).lstrip("0") != str(tic):
            continue
        rows.append({
            "author": str(r["author"]),
            "sector": int(r["sequence_number"]),
            "exptime": float(r["exptime"]),
            "productFilename": str(r["productFilename"]),
            "dataURI": str(r["dataURI"]),
            "size": int(r["size"]) if r["size"] is not np.ma.masked else None,
        })
    return rows


def inventory(tic: int, authors=AUTHORS, cache_dir: Path | None = None,
              max_age_h: float = 24.0, refresh: bool = False, kind: str = "lc") -> tuple[list[dict], dict]:
    """MAST inventory for one TIC, cached as JSON. Falls back to a stale cache if MAST is down."""
    cache_dir = Path(cache_dir or default_cache_dir())
    path = cache_dir / "inventory" / (f"TIC{tic}.json" if kind == "lc" else f"TIC{tic}_{kind}.json")
    cached = json.loads(path.read_text()) if path.exists() else None
    fresh = cached and (time.time() - cached["fetched_unix"]) < max_age_h * 3600
    if cached and fresh and not refresh:
        rows, info = cached["rows"], {"source": "cache", "fetched_at": cached["fetched_at"]}
    else:
        try:
            rows = search_mast(tic, AUTHORS) if kind == "lc" else search_mast(tic, AUTHORS, kind)
            write_json(path, {"tic": tic, "fetched_at": now_et(), "fetched_unix": time.time(), "rows": rows})
            info = {"source": "live", "fetched_at": now_et()}
        except Exception as exc:  # MAST flaked: use what we have, loudly
            if not cached:
                raise RuntimeError(f"MAST search failed for TIC {tic} and no cached inventory: {exc}") from exc
            rows = cached["rows"]
            info = {"source": "stale_cache", "fetched_at": cached["fetched_at"], "error": str(exc)}
    wanted = {a.upper() for a in authors}
    return [r for r in rows if r["author"].upper() in wanted], info


def dedupe(rows: list[dict], exptimes=None, pick: str = "all") -> list[dict]:
    """One product per (author, sector, cadence). pick='best' keeps one per sector.

    When MAST lists two versions of the same product, the lexically last filename wins
    (TESS-SPOC/QLP version tags and SPOC processing dates both sort that way).
    """
    if exptimes:
        keep = {float(e) for e in exptimes}
        rows = [r for r in rows if float(r["exptime"]) in keep]
    by_key: dict[tuple, dict] = {}
    for r in sorted(rows, key=lambda r: r["productFilename"]):
        by_key[(r["author"], r["sector"], float(r["exptime"]))] = r
    out = sorted(by_key.values(), key=lambda r: (r["sector"], AUTHOR_RANK.get(r["author"], 9), r["exptime"]))
    if pick == "best":
        # Lowest author rank; within an author prefer 120 s over 20 s (same photons, 6x smaller).
        best: dict[int, dict] = {}
        for r in out:
            score = (AUTHOR_RANK.get(r["author"], 9), abs(np.log(r["exptime"] / 120.0)))
            if r["sector"] not in best or score < best[r["sector"]][0]:
                best[r["sector"]] = (score, r)
        out = [best[s][1] for s in sorted(best)]
    return out


# ---------------------------------------------------------------- download
def archive_url(uri: str) -> str | None:
    """Direct archive.stsci.edu URL for a MAST TESS/HLSP data URI.

    MAST's download API redirects to an S3 bucket that currently answers 403 from
    this box, so the plain archive mirror is tried first.
    """
    if uri.startswith("mast:HLSP/"):
        return "https://archive.stsci.edu/hlsps/" + uri.split("mast:HLSP/", 1)[1]
    m = re.search(r"-s(\d{4})-(\d{16})-", uri)
    if uri.startswith("mast:TESS/product/") and m:
        sec, ticp = m.groups()
        sub = "/".join(ticp[i:i + 4] for i in range(0, 16, 4))
        return f"https://archive.stsci.edu/missions/tess/tid/s{sec}/{sub}/{uri.rsplit('/', 1)[1]}"
    return None


def _download(uri: str, dest: Path, timeout: float = 300) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tries = [(u, None) for u in [archive_url(uri)] if u] + [(MAST_DOWNLOAD, {"uri": uri})]
    errors = []
    def get(url, params):
        with requests.get(url, params=params, headers=UA, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)

    for url, params in tries:
        try:
            with_backoff(get, url, params)
            with fits.open(tmp) as hdul:  # refuse to cache an HTML error page
                hdul[1].header  # noqa: B018
            tmp.replace(dest)
            return
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError("; ".join(errors))


def _is_good_fits(path: Path) -> bool:
    try:
        with fits.open(path) as hdul:
            return len(hdul) > 1 and "TIME" in hdul[1].columns.names
    except Exception:
        return False


def summarize_file(path: Path) -> dict:
    """Quality summary + crowding keywords from a light-curve (or target-pixel) FITS file."""
    with fits.open(path) as hdul:
        hdr, data = hdul[1].header, hdul[1].data
        names = data.columns.names
        flux_col = next(c for c in ("PDCSAP_FLUX", "KSPSAP_FLUX", "SAP_FLUX", "FLUX") if c in names)
        t = np.asarray(data["TIME"], float)
        q = np.asarray(data["QUALITY"], int) if "QUALITY" in names else np.zeros(len(t), int)
        flux = np.asarray(data[flux_col], float)
        if flux.ndim == 3:  # TPF: a cadence is finite if any pixel is
            flux = np.where(np.isfinite(flux).any(axis=(1, 2)), 1.0, np.nan)
    finite = np.isfinite(t) & np.isfinite(flux)
    good = finite & (q == 0)
    return {
        "n_cadences": int(len(t)), "n_good": int(good.sum()),
        "frac_good": round(float(good.mean()) if len(t) else 0.0, 4),
        "n_finite_flux": int(finite.sum()), "flux_column": flux_col,
        "crowdsap": None if hdr.get("CROWDSAP") is None else float(hdr["CROWDSAP"]),
        "flfrcsap": None if hdr.get("FLFRCSAP") is None else float(hdr["FLFRCSAP"]),
        "t_start_btjd": round(float(np.nanmin(t[np.isfinite(t)])), 5) if np.isfinite(t).any() else None,
        "t_end_btjd": round(float(np.nanmax(t[np.isfinite(t)])), 5) if np.isfinite(t).any() else None,
    }


# ---------------------------------------------------------------- main entry
def fetch(tics, authors=AUTHORS, cache_dir: Path | None = None, exptimes=None, pick: str = "all",
          refresh: bool = False, load: bool = True, max_age_h: float = 24.0, sectors=None, kind: str = "lc"):
    """Fetch light curves for one or more TICs.

    Returns (LightCurveCollection or None, manifest dict). The manifest is also written to
    <cache>/manifests/<key>.json and .csv. `manifest_hash` covers only the product rows,
    so it is identical between a cold and a warm (cached) call.
    """
    cache_dir = Path(cache_dir or default_cache_dir())
    tics = [parse_tic(t) for t in (tics if isinstance(tics, (list, tuple)) else [tics])]
    authors = [a.upper() for a in authors]
    rows_out, inv_info, failures, n_hit, n_dl = [], {}, [], 0, 0
    for tic in tics:
        inv, info = inventory(tic, authors, cache_dir, max_age_h=max_age_h, refresh=refresh, kind=kind)
        inv_info[str(tic)] = {**info, "n_inventory": len(inv)}
        if sectors:
            inv = [r for r in inv if r["sector"] in {int(x) for x in sectors}]
        for r in dedupe(inv, exptimes, pick):
            dest = cache_dir / kind / f"TIC{tic}" / r["productFilename"]
            if dest.exists() and _is_good_fits(dest):
                n_hit += 1
            else:
                try:
                    _download(r["dataURI"], dest)
                    n_dl += 1
                except Exception as exc:
                    failures.append({"tic": tic, "sector": r["sector"], "author": r["author"],
                                     "uri": r["dataURI"], "error": f"{type(exc).__name__}: {exc}"})
                    continue
            s = summarize_file(dest)
            rows_out.append({
                "tic": tic, "sector": r["sector"], "author": r["author"], "cadence_s": r["exptime"],
                "product_uri": r["dataURI"], "filename": r["productFilename"], "local_path": str(dest), **s,
            })
    key = "_".join(f"TIC{t}" for t in tics) + "__" + "-".join(authors) \
        + (f"__exp{'-'.join(str(int(e)) for e in exptimes)}" if exptimes else "") + f"__{pick}" \
        + (f"__s{'-'.join(str(int(x)) for x in sorted(sectors))}" if sectors else "") + ("" if kind == "lc" else f"__{kind}")
    manifest = {
        "tool": "tess_lc_fetch", "generated_at": now_et(), "tics": tics, "authors": authors,
        "exptimes": exptimes, "pick": pick, "kind": kind, "sectors": sectors, "cache_dir": str(cache_dir),
        "inventory": inv_info, "cache_hits": n_hit, "downloaded": n_dl, "failures": failures,
        "complete": not failures, "products": rows_out,
        "manifest_hash": canonical_hash(rows_out),
    }
    mdir = cache_dir / "manifests"
    write_json(mdir / f"{key}.json", manifest)
    with (mdir / f"{key}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(jsonable(rows_out))
    manifest["manifest_paths"] = [str(mdir / f"{key}.json"), str(mdir / f"{key}.csv")]
    return (load_collection(manifest) if (load and kind == "lc") else None), manifest


def load_collection(manifest: dict):
    """LightCurveCollection from a manifest (no quality masking; callers decide)."""
    import lightkurve as lk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lcs = [lk.read(p["local_path"], quality_bitmask="none") for p in manifest["products"]]
    return lk.LightCurveCollection(lcs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tic", nargs="+", help="TIC id(s)")
    ap.add_argument("--author", action="append", choices=AUTHORS, help="repeatable; default all three")
    ap.add_argument("--exptime", action="append", type=float, help="keep only these cadences in s, e.g. 120")
    ap.add_argument("--pick", choices=["all", "best"], default="all", help="best = one product per sector")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--refresh", action="store_true", help="re-query MAST even if the inventory is fresh")
    ap.add_argument("--sectors", type=lambda s: [int(x) for x in s.split(",") if x], help="e.g. 1,4,8")
    ap.add_argument("--tpf", action="store_true", help="fetch target pixel files instead of light curves")
    args = ap.parse_args(argv)
    _, m = fetch(args.tic, authors=args.author or AUTHORS, cache_dir=args.cache_dir,
                 exptimes=args.exptime, pick=args.pick, refresh=args.refresh, load=False,
                 sectors=args.sectors, kind="tpf" if args.tpf else "lc")
    secs = sorted({(p["author"], p["sector"], p["cadence_s"]) for p in m["products"]}, key=lambda x: (x[1], x[0]))
    print(json.dumps({k: m[k] for k in ("tics", "inventory", "cache_hits", "downloaded", "failures",
                                        "complete", "manifest_hash", "manifest_paths")}, indent=2))
    print(f"{len(secs)} products:", ", ".join(f"{a} s{s} {int(c)}s" for a, s, c in secs))
    return 0 if m["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
