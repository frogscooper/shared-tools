"""catalog_veto: is this TIC already a known planet, TOI, CTOI, or eclipsing binary?

Sources (all public, read-only):
  exofop_toi, exofop_ctoi  ExoFOP CSV dumps, cached per day as <name>_YYYY-MM-DD.csv
  nasa_pscomppars          NASA Exoplanet Archive TAP, table pscomppars (by tic_id)
  prsa_tess_eb             Prsa+2022 TESS EB catalog (J/ApJS/258/16) on the CfA VizieR mirror
                           (CDS VizieR 404s), cone of --eb-radius around the target
  gaia_dr3_neighbors       Gaia DR3 gaia_source cone of --neighbor-radius (default 2 arcmin)
  gaia_dr3_eb              Gaia DR3 vari_eclipsing_binary within --eb-radius (default 21 arcsec)

Fail closed: every source reports status ok / empty / down. If any source is down the
verdict is never "clean", and a failed query is never read as "not in catalog".
The last good result per source is kept with its date and shown next to a failure.

    catalog-veto 261136679
    catalog-veto 288183107 --period 12.96393 --epoch 3319.20707 --depth 2140
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .common import default_cache_dir, now_et, parse_tic, today_et, write_json

TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
CTOI_URL = "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv"
NASA_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
VIZIER_CFA = "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv"
PRSA_EB = "J/ApJS/258/16/tess-ebs"
MAST_TIC = "https://mast.stsci.edu/api/v0/invoke"
UA = {"User-Agent": "shared-tools catalog_veto (read-only research use)"}
KNOWN_DISPOSITIONS = {"KP", "CP"}  # known planet / confirmed planet in TOI dispositions
FP_DISPOSITIONS = {"FP", "FA", "EB"}


class SourceDown(RuntimeError):
    """Raised when a catalog query fails. Never interpreted as 'no match'."""


# ---------------------------------------------------------------- HTTP (one seam for tests)
RETRIES, BACKOFF_S = 2, 2.0


def http(method: str, url: str, timeout=(20, 120), **kw) -> requests.Response:
    """One HTTP call with a retry. Raises SourceDown on any failure."""
    err = None
    for attempt in range(RETRIES):
        try:
            return _http_once(method, url, timeout, **kw)
        except (SourceDown, requests.RequestException) as exc:
            err = exc
            if attempt + 1 < RETRIES:
                time.sleep(BACKOFF_S * (attempt + 1))
    raise err if isinstance(err, SourceDown) else SourceDown(f"{type(err).__name__}: {err}")


def _http_once(method, url, timeout, **kw) -> requests.Response:
    r = requests.request(method, url, headers=UA, timeout=timeout, **kw)
    if r.status_code >= 400:
        title = ""
        if "<title>" in r.text.lower():
            low = r.text.lower()
            title = r.text[low.index("<title>") + 7: low.index("</title>")].strip() if "</title>" in low else ""
        raise SourceDown(f"HTTP {r.status_code} from {url.split('?')[0]}" + (f" ({title})" if title else ""))
    head = r.text[:300].lstrip().lower()
    if head.startswith("<!doctype html") or head.startswith("<html"):
        raise SourceDown(f"HTML page instead of data from {url.split('?')[0]}")
    return r


def _num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _digits(v) -> str:
    s = "".join(ch for ch in str(v) if ch.isdigit())
    return s.lstrip("0") or ("0" if s else "")


# ---------------------------------------------------------------- per-source cache
class SourceCache:
    """Last good result per (tic, source), kept with its fetch time."""

    def __init__(self, cache_dir: Path, tic: int):
        self.dir = Path(cache_dir) / "catalog_veto" / f"TIC{tic}"

    def get(self, name: str):
        p = self.dir / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, name: str, payload: dict) -> None:
        write_json(self.dir / f"{name}.json", {**payload, "fetched_at": now_et(), "fetched_unix": time.time()})


def run_source(name: str, fn, cache: SourceCache, max_age_h: float, refresh: bool) -> dict:
    """Run one source with fail-closed bookkeeping. fn() -> list of match dicts."""
    last = cache.get(name)
    if last and not refresh and time.time() - last["fetched_unix"] < max_age_h * 3600:
        return {"status": "empty" if not last["matches"] else "ok", "matches": last["matches"],
                "fetched_at": last["fetched_at"], "from_cache": True, "error": None,
                **({"extra": last["extra"]} if "extra" in last else {})}
    try:
        res = fn()
        matches, extra = (res if isinstance(res, tuple) else (res, None))
    except Exception as exc:  # fail closed
        return {"status": "down", "matches": [], "error": f"{type(exc).__name__}: {exc}",
                "failed_at": now_et(), "from_cache": False,
                "last_good": None if not last else {"fetched_at": last["fetched_at"], "matches": last["matches"]}}
    cache.put(name, {"matches": matches, **({"extra": extra} if extra is not None else {})})
    out = {"status": "ok" if matches else "empty", "matches": matches, "fetched_at": now_et(),
           "from_cache": False, "error": None}
    if extra is not None:
        out["extra"] = extra
    return out


# ---------------------------------------------------------------- sources
def exofop_table(name: str, url: str, cache_dir: Path, refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Today's ExoFOP dump (date-stamped cache). Raises SourceDown if today's copy can't be had."""
    d = Path(cache_dir) / "catalogs"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}_{today_et()}.csv"
    if refresh or not (path.exists() and path.stat().st_size > 1000):
        r = http("GET", url, timeout=(20, 300))
        if "TIC ID" not in r.text[:2000]:
            raise SourceDown(f"{name}: response has no 'TIC ID' header")
        tmp = path.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(path)
    return pd.read_csv(path, comment="#", low_memory=False), str(path)


def last_dated_copy(name: str, cache_dir: Path) -> str | None:
    files = sorted((Path(cache_dir) / "catalogs").glob(f"{name}_20*.csv"))
    return str(files[-1]) if files else None


def exofop_matches(df: pd.DataFrame, tic: int, kind: str) -> list[dict]:
    hit = df[df["TIC ID"].map(_digits) == str(tic)]
    out = []
    for _, r in hit.iterrows():
        if kind == "toi":
            out.append({"toi": str(r.get("TOI")), "tess_disposition": r.get("TESS Disposition"),
                        "tfopwg_disposition": r.get("TFOPWG Disposition"),
                        "period_d": _num(r.get("Period (days)")), "epoch_bjd": _num(r.get("Epoch (BJD)")),
                        "duration_h": _num(r.get("Duration (hours)")), "depth_ppm": _num(r.get("Depth (ppm)")),
                        "planet_name": None if pd.isna(r.get("Planet Name")) else r.get("Planet Name"),
                        "ra": r.get("RA"), "dec": r.get("Dec"), "comments": None if pd.isna(r.get("Comments")) else r.get("Comments")})
        else:
            out.append({"ctoi": str(r.get("CTOI")), "promoted_to_toi": None if pd.isna(r.get("Promoted to TOI")) else str(r.get("Promoted to TOI")),
                        "user_disposition": r.get("User Disposition"), "tfopwg_disposition": r.get("TFOPWG Disposition"),
                        "period_d": _num(r.get("Period (days)")), "epoch_bjd": _num(r.get("Transit Epoch (BJD)")),
                        "duration_h": _num(r.get("Duration (hrs)")), "depth_ppm": _num(r.get("Depth ppm")),
                        "notes": None if pd.isna(r.get("Notes")) else r.get("Notes")})
    return [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in m.items()} for m in out]


def nasa_pscomppars(tic: int) -> list[dict]:
    q = ("SELECT pl_name, hostname, tic_id, pl_orbper, pl_tranmid, pl_trandur, pl_trandep, "
         f"discoverymethod, disc_year FROM pscomppars WHERE tic_id = 'TIC {int(tic)}'")
    r = http("GET", NASA_TAP, params={"query": q, "format": "csv"})
    if not r.text.lstrip().startswith("pl_name"):
        raise SourceDown(f"NASA TAP returned unexpected body: {r.text[:200]!r}")
    df = pd.read_csv(io.StringIO(r.text))
    return [{"pl_name": x.pl_name, "period_d": _num(x.pl_orbper), "tranmid_bjd": _num(x.pl_tranmid),
             "duration_h": _num(x.pl_trandur), "depth_pct": _num(x.pl_trandep),
             "discoverymethod": x.discoverymethod, "disc_year": _num(x.disc_year)} for x in df.itertuples()]


def gaia_tap(adql: str) -> pd.DataFrame:
    r = http("POST", GAIA_TAP, timeout=(20, 180),
             data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql})
    if "ERROR" in r.text[:200] or "," not in r.text.splitlines()[0]:
        raise SourceDown(f"Gaia TAP error: {r.text[:200]!r}")
    return pd.read_csv(io.StringIO(r.text))


GAIA_EPOCH_SHIFT_YR = 16.0  # Gaia DR3 positions are epoch 2016.0; TIC positions are epoch 2000.0
POS_MARGIN_ARCSEC = 30.0     # widen cones so high-proper-motion stars are not lost


def _sep_j2000(df: pd.DataFrame, ra: float, dec: float) -> np.ndarray:
    """Separation (arcsec) of Gaia sources moved back to epoch 2000, from a TIC (J2000) position."""
    pmra = df["pmra"].fillna(0).to_numpy(float)
    pmdec = df["pmdec"].fillna(0).to_numpy(float)
    dec0 = df["dec"].to_numpy(float) - pmdec * GAIA_EPOCH_SHIFT_YR / 3.6e6
    ra0 = df["ra"].to_numpy(float) - pmra * GAIA_EPOCH_SHIFT_YR / 3.6e6 / np.cos(np.radians(dec0))
    dra = (ra0 - ra) * np.cos(np.radians(dec))
    return np.hypot(dra, dec0 - dec) * 3600.0


def _cone(table_sql: str, cols: str, ra: float, dec: float, radius_arcsec: float) -> pd.DataFrame:
    r = (radius_arcsec + POS_MARGIN_ARCSEC) / 3600.0
    adql = (f"SELECT {cols} FROM {table_sql} WHERE 1=CONTAINS(POINT('ICRS', g.ra, g.dec), "
            f"CIRCLE('ICRS', {ra}, {dec}, {r}))")
    try:
        df = gaia_tap(adql)
    except SourceDown as exc:  # Gaia archive down (e.g. DR4 downtime): same table from the VizieR mirror
        try:
            return _cone_vizier(table_sql, ra, dec, radius_arcsec)
        except SourceDown as exc2:
            raise SourceDown(f"Gaia TAP: {exc}; VizieR mirror: {exc2}") from exc2
    if df.empty:
        return df
    df["source_id"] = df["source_id"].astype("int64")
    df["sep_arcsec"] = _sep_j2000(df, ra, dec)
    return df[df["sep_arcsec"] <= radius_arcsec].sort_values("sep_arcsec").reset_index(drop=True)


VIZIER_GAIA = {  # VizieR mirrors of the Gaia DR3 tables (used when the Gaia archive is down, e.g. DR4 downtime)
    "gaiadr3.gaia_source": ("I/355/gaiadr3", {"Source": "source_id", "RA_ICRS": "ra", "DE_ICRS": "dec", "pmRA": "pmra",
                                              "pmDE": "pmdec", "Gmag": "phot_g_mean_mag", "Plx": "parallax",
                                              "RUWE": "ruwe"}),
    "gaiadr3.vari_eclipsing_binary": ("I/358/veb", {"Source": "source_id", "Freq": "frequency", "RA_ICRS": "ra",
                                                    "DE_ICRS": "dec"}),
}
NSS_CATALOG = "I/357"            # Gaia DR3 non-single stars: every nss_two_body_orbit model + acceleration + trends
NSS_BINARY_MODELS = ("SB2", "EclipsingBinary", "EclipsingSpectro")   # models that by themselves mean "binary star"
RUWE_HIGH = 1.4


def parse_vizier_multi(text: str) -> list[tuple[str, pd.DataFrame]]:
    """ASU-TSV with several #Table blocks -> [(table name, DataFrame)] (empty tables dropped)."""
    if "#INFO" not in text and "#RESOURCE" not in text:
        raise SourceDown(f"VizieR response is not ASU-TSV: {text[:200]!r}")
    bad = [l for l in text.splitlines() if l.startswith("#***") or (l.startswith("#INFO") and "rror" in l)]
    if bad:
        raise SourceDown("VizieR reported an error: " + " | ".join(bad)[:400])
    out, name, block = [], None, []

    def flush():
        lines = [l for l in block if l and not l.startswith("#")]
        if name and len(lines) >= 4:
            header = lines[0].split("\t")
            out.append((name, pd.DataFrame([l.split("\t") for l in lines[3:]], columns=header).apply(lambda c: c.str.strip())))
    for line in text.splitlines():
        if line.startswith("#Name: "):
            flush()
            name, block = line[7:].strip(), []
        else:
            block.append(line)
    flush()
    return out


def vizier_cone(catalog: str, cols: list[str], ra: float, dec: float, radius_arcsec: float) -> list[tuple[str, pd.DataFrame]]:
    params = {"-source": catalog, "-c": f"{ra} {dec}", "-c.rs": radius_arcsec, "-out": ",".join(cols),
              "-out.add": "_r", "-out.max": 5000}
    return parse_vizier_multi(http("GET", VIZIER_CFA, params=params).text)


def _cone_vizier(table_sql: str, ra: float, dec: float, radius_arcsec: float) -> pd.DataFrame:
    base = table_sql.split(" AS ")[0].split(" JOIN ")[0].strip()
    cat, colmap = VIZIER_GAIA[base]
    tabs = [t for t in vizier_cone(cat, list(colmap), ra, dec, radius_arcsec + POS_MARGIN_ARCSEC) if t[0] == cat]
    if not tabs:
        return pd.DataFrame(columns=list(colmap.values()) + ["sep_arcsec"])
    df = tabs[0][1].rename(columns=colmap)
    for c in df.columns:
        if c != "source_id":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "phot_g_mean_mag" not in df and base == "gaiadr3.vari_eclipsing_binary":
        g = _cone_vizier("gaiadr3.gaia_source", ra, dec, radius_arcsec)
        df = df.merge(g[["source_id", "phot_g_mean_mag", "pmra", "pmdec"]].astype({"source_id": str}), on="source_id", how="left")
    for c in ("pmra", "pmdec"):
        if c not in df:
            df[c] = 0.0
    df["source_id"] = df["source_id"].astype("int64")
    df["sep_arcsec"] = _sep_j2000(df, ra, dec)
    return df[df["sep_arcsec"] <= radius_arcsec].sort_values("sep_arcsec").reset_index(drop=True)


def gaia_nss(ra: float, dec: float, radius_arcsec: float = 5.0, gaia_id=None) -> tuple[list[dict], dict]:
    """Gaia DR3 non-single-star solutions (all I/357 tables, one VizieR call) + the target's RUWE."""
    rows = []
    for table, df in vizier_cone(NSS_CATALOG, ["Source", "NSSmodel", "Per", "ecc"], ra, dec, radius_arcsec):
        if not table.startswith(NSS_CATALOG + "/"):
            continue
        for _, x in df.iterrows():
            sid = _digits(x.get("Source"))
            sep = _num(x.get("_r"))
            rows.append({"table": table, "source_id": int(sid) if sid else None, "model": x.get("NSSmodel") or table.split("/")[-1],
                         "period_d": _num(x.get("Per")), "ecc": _num(x.get("ecc")), "sep_arcsec": sep,
                         "is_target": (gaia_id is not None and sid == str(gaia_id)) or (sep is not None and sep <= 1.5)})
    ruwe = None
    try:
        g = _cone_vizier("gaiadr3.gaia_source", ra, dec, 3.0)
        if len(g):
            tid = pick_target(g, gaia_id)
            if tid is not None:
                ruwe = _num(g.loc[g["source_id"] == tid, "ruwe"].iloc[0])
    except SourceDown:
        pass
    return rows, {"target_ruwe": ruwe}


def pick_target(df: pd.DataFrame, gaia_id=None):
    """Target row: TIC's Gaia id if present (DR2 ids mostly carry over to DR3), else brightest within 2"."""
    if df.empty:
        return None
    if gaia_id is not None and (df["source_id"] == int(gaia_id)).any():
        return int(gaia_id)
    near = df[df["sep_arcsec"] <= 2.0]
    if near.empty:
        return None
    return int(near.sort_values("phot_g_mean_mag")["source_id"].iloc[0])


def gaia_neighbors(ra: float, dec: float, radius_arcsec: float, depth_ppm=None, gaia_id=None) -> tuple[list[dict], dict]:
    df = _cone("gaiadr3.gaia_source AS g",
               "g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.phot_g_mean_mag, g.parallax, g.ruwe",
               ra, dec, radius_arcsec)
    tid = pick_target(df, gaia_id)
    if tid is None:
        return ([] if df.empty else _neighbor_rows(df, None, None, depth_ppm)), {"target_source_id": None, "target_gmag": None}
    g0 = _num(df.loc[df["source_id"] == tid, "phot_g_mean_mag"].iloc[0])
    return _neighbor_rows(df, tid, g0, depth_ppm), {"target_source_id": tid, "target_gmag": g0}


def _neighbor_rows(df, tid, g0, depth_ppm):
    rows = []
    for x in df.itertuples():
        g = _num(x.phot_g_mean_mag)
        dg = None if (g is None or g0 is None) else round(g - g0, 3)
        row = {"source_id": int(x.source_id), "sep_arcsec": round(float(x.sep_arcsec), 2), "gmag": g,
               "delta_g": dg, "parallax": _num(x.parallax), "ruwe": _num(x.ruwe),
               "is_target": tid is not None and int(x.source_id) == tid}
        if depth_ppm and dg is not None and not row["is_target"]:
            # Eclipse depth this neighbor would need to produce the observed depth when fully blended.
            from .secondary_dilution import needed_depth

            row["needed_eclipse_depth"] = round(needed_depth(depth_ppm, dg), 4)
            row["could_mimic_depth"] = row["needed_eclipse_depth"] <= 1.0
        rows.append(row)
    return rows


def gaia_ebs(ra: float, dec: float, radius_arcsec: float, gaia_id=None) -> list[dict]:
    df = _cone("gaiadr3.vari_eclipsing_binary AS eb JOIN gaiadr3.gaia_source AS g ON eb.source_id = g.source_id",
               "eb.source_id, eb.frequency, g.ra, g.dec, g.pmra, g.pmdec, g.phot_g_mean_mag", ra, dec, radius_arcsec)
    return [{"source_id": int(x.source_id), "sep_arcsec": round(float(x.sep_arcsec), 2),
             "gmag": _num(x.phot_g_mean_mag),
             "is_target": (gaia_id is not None and int(x.source_id) == int(gaia_id)) or float(x.sep_arcsec) <= 1.0,
             # Gaia's frequency is the eclipse-model frequency; 1/f is the orbital period.
             "period_d": (1.0 / x.frequency) if _num(x.frequency) else None} for x in df.itertuples()]


def parse_vizier_tsv(text: str) -> pd.DataFrame:
    if "#INFO" not in text and "#RESOURCE" not in text:
        raise SourceDown(f"VizieR response is not ASU-TSV: {text[:200]!r}")
    bad = [l for l in text.splitlines() if l.startswith("#***") or (l.startswith("#INFO") and "rror" in l)]
    if bad:
        raise SourceDown("VizieR reported an error: " + " | ".join(bad)[:400])
    lines = [l for l in text.splitlines() if l and not l.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame()
    header = lines[0].split("\t")
    data = [l.split("\t") for l in lines[3:]]  # skip units and dashes rows
    return pd.DataFrame(data, columns=header).apply(lambda c: c.str.strip())


def prsa_ebs(tic: int, ra: float, dec: float, radius_arcsec: float) -> list[dict]:
    params = {"-source": PRSA_EB, "-c": f"{ra} {dec}", "-c.rs": radius_arcsec,
              "-out": "TIC,Per,BJD0,Morph,Dp-pf,Ds-pf,Sectors", "-out.add": "_r"}
    for attempt in range(RETRIES):  # VizieR sometimes answers 200 with an error block
        try:
            df = parse_vizier_tsv(http("GET", VIZIER_CFA, params=params).text)
            break
        except SourceDown:
            if attempt + 1 == RETRIES:
                raise
            time.sleep(BACKOFF_S)
    return [{"tic": int(x["TIC"]), "sep_arcsec": _num(x["_r"]), "period_d": _num(x["Per"]),
             "bjd0": _num(x["BJD0"]), "morph": _num(x["Morph"]), "is_target": _digits(x["TIC"]) == str(tic),
             "sectors": x.get("Sectors")} for _, x in df.iterrows()]


def tic_position(tic: int) -> dict:
    """RA/Dec from the MAST TIC (astroquery)."""
    from astroquery.mast import Catalogs

    t = Catalogs.query_criteria(catalog="TIC", ID=int(tic))
    if t is None or len(t) == 0:
        raise SourceDown(f"TIC {tic} not found in MAST TIC")
    gaia = t["GAIA"][0] if "GAIA" in t.colnames else None
    gaia = None if (gaia is None or np.ma.is_masked(gaia) or not str(gaia).strip()) else int(gaia)
    return {"ra": float(t["ra"][0]), "dec": float(t["dec"][0]), "tmag": _num(t["Tmag"][0]), "gaia_dr2_id": gaia,
            "pmra": _num(t["pmRA"][0]) if "pmRA" in t.colnames else None,
            "pmdec": _num(t["pmDEC"][0]) if "pmDEC" in t.colnames else None}


# ---------------------------------------------------------------- period aliases
def period_relation(p: float, p_cat: float, tol: float = 0.01, max_n: int = 5) -> dict | None:
    """Does p equal n*p_cat or p_cat/n (n<=max_n) within fractional tol?"""
    if not p or not p_cat:
        return None
    for n in range(1, max_n + 1):
        for ratio, label in ((n, f"{n}x"), (1.0 / n, f"1/{n}x")):
            if abs(p / (p_cat * ratio) - 1) <= tol:
                return {"relation": "same" if n == 1 else label, "ratio": round(p / p_cat, 5), "catalog_period_d": p_cat}
    return None


def epoch_offset(p, t0_btjd, cat_p, cat_t0_bjd, duration_h):
    """Phase offset of our epoch vs a catalog epoch, in hours (folded onto the catalog period)."""
    if None in (p, t0_btjd, cat_p, cat_t0_bjd):
        return None
    cat_t0 = cat_t0_bjd - 2457000.0 if cat_t0_bjd > 2400000 else cat_t0_bjd
    d = ((t0_btjd - cat_t0 + 0.5 * cat_p) % cat_p) - 0.5 * cat_p
    return round(d * 24.0, 3)


# ---------------------------------------------------------------- main
def veto(tic, ra=None, dec=None, period_d=None, epoch_btjd=None, depth_ppm=None,
         eb_radius_arcsec: float = 21.0, neighbor_radius_arcsec: float = 120.0,
         cache_dir: Path | None = None, max_age_h: float = 24.0, refresh: bool = False,
         alias_tol: float = 0.01, nss: bool = True) -> dict:
    tic = parse_tic(tic)
    cache_dir = Path(cache_dir or default_cache_dir())
    cache = SourceCache(cache_dir, tic)
    src: dict[str, dict] = {}

    for name, url, kind in (("exofop_toi", TOI_URL, "toi"), ("exofop_ctoi", CTOI_URL, "ctoi")):
        try:
            df, path = exofop_table(kind, url, cache_dir, refresh)
            m = exofop_matches(df, tic, kind)
            src[name] = {"status": "ok" if m else "empty", "matches": m, "cache_file": path, "error": None}
        except Exception as exc:
            last = last_dated_copy(kind, cache_dir)
            stale = None
            if last:
                stale = {"file": last, "matches": exofop_matches(pd.read_csv(last, comment="#", low_memory=False), tic, kind)}
            src[name] = {"status": "down", "matches": [], "error": f"{type(exc).__name__}: {exc}",
                         "failed_at": now_et(), "last_good": stale}

    gaia_id = None
    if ra is None or dec is None:
        pos = next((m for m in src.get("exofop_toi", {}).get("matches", []) if m.get("ra")), None)
        try:
            p = tic_position(tic)
            ra, dec, gaia_id = p["ra"], p["dec"], p["gaia_dr2_id"]
            src["tic_position"] = {"status": "ok", "ra": ra, "dec": dec, "tmag": p["tmag"],
                                   "gaia_dr2_id": gaia_id, "error": None}
        except Exception as exc:
            src["tic_position"] = {"status": "down", "error": f"{type(exc).__name__}: {exc}", "failed_at": now_et()}
            if pos:
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                c = SkyCoord(pos["ra"], pos["dec"], unit=(u.hourangle, u.deg))
                ra, dec = c.ra.deg, c.dec.deg

    src["nasa_pscomppars"] = run_source("nasa_pscomppars", lambda: nasa_pscomppars(tic), cache, max_age_h, refresh)
    if ra is None:
        for name in ("prsa_tess_eb", "gaia_dr3_neighbors", "gaia_dr3_eb", "gaia_dr3_nss"):
            src[name] = {"status": "down", "matches": [], "error": "no target coordinates", "failed_at": now_et()}
    else:
        src["prsa_tess_eb"] = run_source(f"prsa_tess_eb_{eb_radius_arcsec:g}",
                                         lambda: prsa_ebs(tic, ra, dec, eb_radius_arcsec), cache, max_age_h, refresh)
        src["gaia_dr3_neighbors"] = run_source(
            f"gaia_dr3_neighbors_{neighbor_radius_arcsec:g}_{depth_ppm}",
            lambda: gaia_neighbors(ra, dec, neighbor_radius_arcsec, depth_ppm, gaia_id), cache, max_age_h, refresh)
        src["gaia_dr3_eb"] = run_source(f"gaia_dr3_eb_{eb_radius_arcsec:g}",
                                        lambda: gaia_ebs(ra, dec, eb_radius_arcsec, gaia_id), cache, max_age_h, refresh)
        if nss:
            src["gaia_dr3_nss"] = run_source("gaia_dr3_nss_5", lambda: gaia_nss(ra, dec, 5.0, gaia_id), cache, max_age_h,
                                             refresh)

    # ---- flags
    tois = src["exofop_toi"]["matches"]
    ctois = src["exofop_ctoi"]["matches"]
    flags = {
        "known_planet": bool(src["nasa_pscomppars"]["matches"]) or any(
            str(m.get("tfopwg_disposition")) in KNOWN_DISPOSITIONS or str(m.get("tess_disposition")) in KNOWN_DISPOSITIONS
            for m in tois),
        "toi": bool(tois),
        "toi_false_positive": any(str(m.get("tfopwg_disposition")) in FP_DISPOSITIONS for m in tois),
        "ctoi": bool(ctois),
        "eb": any(m["is_target"] for m in src.get("prsa_tess_eb", {}).get("matches", []))
              or any(m["is_target"] for m in src.get("gaia_dr3_eb", {}).get("matches", [])),
        "nearby_eb": any(not m["is_target"] for m in src.get("prsa_tess_eb", {}).get("matches", []))
                     or any(not m["is_target"] for m in src.get("gaia_dr3_eb", {}).get("matches", [])),
    }
    nss_t = [m for m in src.get("gaia_dr3_nss", {}).get("matches", []) if m["is_target"]]
    ruwe = (src.get("gaia_dr3_nss", {}).get("extra") or {}).get("target_ruwe")
    flags["nss_orbit"] = bool(nss_t)
    flags["nss_binary"] = any(m["model"] in NSS_BINARY_MODELS for m in nss_t)
    flags["high_ruwe"] = ruwe is not None and ruwe > RUWE_HIGH
    nb = src.get("gaia_dr3_neighbors", {}).get("matches", [])
    flags["bright_neighbor_within_eb_radius"] = any(
        (not m["is_target"]) and m["sep_arcsec"] <= eb_radius_arcsec and m["delta_g"] is not None and m["delta_g"] < 5
        for m in nb)

    # ---- period aliases against every catalog period we found
    aliases = []
    if period_d:
        cands = [("exofop_toi", m.get("toi"), m.get("period_d"), m.get("epoch_bjd"), m.get("duration_h")) for m in tois]
        cands += [("exofop_ctoi", m.get("ctoi"), m.get("period_d"), m.get("epoch_bjd"), m.get("duration_h")) for m in ctois]
        cands += [("nasa_pscomppars", m["pl_name"], m["period_d"], m["tranmid_bjd"], m["duration_h"])
                  for m in src["nasa_pscomppars"]["matches"]]
        cands += [("prsa_tess_eb", m["tic"], m["period_d"], m["bjd0"], None) for m in src.get("prsa_tess_eb", {}).get("matches", [])]
        cands += [("gaia_dr3_eb", m["source_id"], m["period_d"], None, None) for m in src.get("gaia_dr3_eb", {}).get("matches", [])]
        cands += [("gaia_dr3_nss", f"{m['source_id']} {m['model']}", m["period_d"], None, None) for m in nss_t if m["period_d"]]
        for source, ident, p_cat, t0_cat, dur in cands:
            rel = period_relation(period_d, p_cat, alias_tol)
            if rel:
                rel.update({"source": source, "id": ident,
                            "epoch_offset_h": epoch_offset(period_d, epoch_btjd, p_cat, t0_cat, dur)})
                aliases.append(rel)
    flags["period_matches_catalog"] = bool(aliases)

    down = [k for k, v in src.items() if v["status"] == "down"]
    hit = [k for k in ("known_planet", "toi", "ctoi", "eb", "nearby_eb", "nss_binary", "period_matches_catalog") if flags[k]]
    if hit:
        verdict = "known_object"
    elif down:
        verdict = "incomplete"
    else:
        verdict = "clean"
    reason = (f"flags: {', '.join(hit)}" if hit else "no catalog matches") + \
             (f"; sources down: {', '.join(down)} (a failed query is not a non-match)" if down else "")
    notes = []
    if flags["known_planet"] and flags["eb"]:
        notes.append("target is a known planet and also has an EB-catalog entry (Prsa+2022 and Gaia EB lists "
                     "contain some hot Jupiters); read eb as 'catalog entry exists', not as a disproof")
    if flags["nss_orbit"] and not flags["nss_binary"]:
        notes.append("Gaia DR3 has a non-single-star solution for the target (" + ", ".join(
            f"{m['model']}" + (f" P={m['period_d']:.2f} d" if m["period_d"] else "") for m in nss_t) +
            "): a companion exists; check the transit is not on it")
    if flags["high_ruwe"]:
        notes.append(f"target RUWE {ruwe:.2f} > {RUWE_HIGH}: astrometry suggests an unresolved companion")
    if not nb or not any(m["is_target"] for m in nb):
        notes.append("Gaia target counterpart not identified; delta_g columns are empty")
    result = {
        "tool": "catalog_veto", "tic": tic, "queried_at": now_et(), "ra": ra, "dec": dec,
        "inputs": {"period_d": period_d, "epoch_btjd": epoch_btjd, "depth_ppm": depth_ppm,
                   "eb_radius_arcsec": eb_radius_arcsec, "neighbor_radius_arcsec": neighbor_radius_arcsec},
        "summary": {"verdict": verdict, "clean": verdict == "clean", "complete": not down,
                    "sources_down": down, "reason": reason, "notes": notes},
        "flags": flags, "period_aliases": aliases, "sources": src,
    }
    write_json(cache_dir / "catalog_veto" / f"TIC{tic}" / "result.json", result)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tic")
    ap.add_argument("--ra", type=float)
    ap.add_argument("--dec", type=float)
    ap.add_argument("--period", type=float, help="trial period, days")
    ap.add_argument("--epoch", type=float, help="trial epoch, BTJD")
    ap.add_argument("--depth", type=float, help="trial depth, ppm (enables neighbor could-mimic column)")
    ap.add_argument("--eb-radius", type=float, default=21.0, help="arcsec, Prsa + Gaia EB match (default 21)")
    ap.add_argument("--neighbor-radius", type=float, default=120.0, help="arcsec, Gaia neighbor table (default 120)")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--refresh", action="store_true", help="ignore per-source cache, query live")
    ap.add_argument("--out", type=Path, help="write full JSON here")
    ap.add_argument("--full", action="store_true", help="print the full JSON instead of the summary")
    a = ap.parse_args(argv)
    r = veto(a.tic, a.ra, a.dec, a.period, a.epoch, a.depth, a.eb_radius, a.neighbor_radius, a.cache_dir, refresh=a.refresh)
    if a.out:
        write_json(a.out, r)
    if a.full:
        print(json.dumps(r, indent=2, default=str))
    else:
        brief = {k: {"status": v["status"], "n": len(v.get("matches", [])), "error": v.get("error")}
                 for k, v in r["sources"].items()}
        print(json.dumps({"tic": r["tic"], "summary": r["summary"], "flags": r["flags"],
                          "period_aliases": r["period_aliases"], "sources": brief}, indent=2, default=str))
    return {"clean": 0, "known_object": 1, "incomplete": 3}[r["summary"]["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
