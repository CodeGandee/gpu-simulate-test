from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gpu_simulate_test.io import assert_columns, read_csv, read_json, stable_id, write_csv, write_json


def test_assert_columns_missing_raises() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="missing required columns"):
        assert_columns(df, ["a", "b"], context="unit")


def test_write_read_csv_validates_columns(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    df = pd.DataFrame({"a": [1], "b": [2]})
    write_csv(path, df, required_columns=["a", "b"])

    out = read_csv(path, required_columns=["a"], context="read")
    assert list(out.columns) == ["a", "b"]


def test_write_json_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    write_json(path, {"b": 1, "a": 2})
    data = read_json(path)
    assert data == {"a": 2, "b": 1}

    raw = path.read_text(encoding="utf-8")
    # sorted keys + trailing newline
    assert raw.endswith("\n")
    assert raw.index('"a"') < raw.index('"b"')


def test_stable_id_is_deterministic() -> None:
    a = stable_id(["hello", "world"], prefix="x", length=12)
    b = stable_id(["hello", "world"], prefix="x", length=12)
    c = stable_id(["hello", "WORLD"], prefix="x", length=12)
    assert a == b
    assert a != c

