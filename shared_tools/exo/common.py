"""Small helpers shared by the exo tools."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ET = ZoneInfo("America/New_York")


def default_cache_dir() -> Path:
    """$SHARED_TOOLS_CACHE, else ~/.cache/shared_tools."""
    return Path(os.environ.get("SHARED_TOOLS_CACHE", Path.home() / ".cache" / "shared_tools"))


def now_et() -> str:
    return datetime.now(ET).isoformat(timespec="seconds")


def today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def parse_tic(value) -> int:
    """Accept 261136679, '261136679', 'TIC 261136679', 'TIC261136679'."""
    digits = re.findall(r"\d+", str(value))
    if not digits:
        raise ValueError(f"cannot parse a TIC id from {value!r}")
    return int(digits[0])


def jsonable(obj):
    """Turn numpy / astropy scalars and NaN into plain JSON values."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not math.isfinite(x) else x
    if isinstance(obj, Path):
        return str(obj)
    if obj is np.ma.masked:
        return None
    return obj


def canonical_hash(obj) -> str:
    """sha256 of sorted-key JSON. Same content -> same hash, regardless of dict order."""
    blob = json.dumps(jsonable(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def write_json(path: Path, obj) -> Path:
    """Atomic write so a crash never leaves half a file in the cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=False) + "\n")
    tmp.replace(path)
    return path
