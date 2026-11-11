"""Access to the assumptions register (registers/assumptions.yaml).

The register is the single source of the default numbers used by the code, so every number shown
to a user can be traced to a source or is flagged as an estimate.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

REGISTER = Path(__file__).resolve().parents[2] / "registers" / "assumptions.yaml"


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, dict]:
    rows = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {r["name"]: r for r in rows}


def entry(name: str, path: str | Path = REGISTER) -> dict:
    return _load(str(path))[name]


def central(name: str, path: str | Path = REGISTER) -> float:
    return float(entry(name, path)["central"])


def low_high(name: str, path: str | Path = REGISTER) -> tuple[float, float]:
    e = entry(name, path)
    return float(e["low"]), float(e["high"])


def all_entries(path: str | Path = REGISTER) -> list[dict]:
    return list(_load(str(path)).values())
