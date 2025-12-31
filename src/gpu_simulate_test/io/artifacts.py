from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


def assert_columns(df: pd.DataFrame, required: Sequence[str], *, context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


def read_csv(path: Path, *, required_columns: Sequence[str], context: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert_columns(df, required_columns, context=context)
    return df


def write_csv(path: Path, df: pd.DataFrame, *, required_columns: Sequence[str]) -> None:
    assert_columns(df, required_columns, context=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def stable_id(parts: Iterable[str], *, prefix: str, length: int = 10) -> str:
    payload = "\x1f".join([prefix, *parts]).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:length]
    return f"{prefix}_{digest}"

