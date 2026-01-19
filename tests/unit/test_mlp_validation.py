from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gpu_simulate_test.vidur_ext.mlp_validation import MlpCsvValidationError, validate_mlp_csv


def test_validate_mlp_csv_nan_policy_auto_strict_reject_non_strict_drop(tmp_path: Path) -> None:
    csv_path = tmp_path / "mlp_missing.csv"
    df = pd.DataFrame(
        {
            "num_tokens": [128, 256],
            "time_stats.op.min": [1.0, 2.0],
            "time_stats.op.max": [1.0, 2.0],
            "time_stats.op.mean": [1.0, 2.0],
            "time_stats.op.median": [None, 2.0],
        }
    )
    df.to_csv(csv_path, index=False)

    with pytest.raises(MlpCsvValidationError):
        validate_mlp_csv(csv_path, mode="strict", nan_policy="auto")

    result = validate_mlp_csv(csv_path, mode="non_strict", nan_policy="auto")
    assert result.missing_cells_total > 0
    assert result.nan_policy_effective == "drop"
    assert result.warnings


def test_validate_mlp_csv_nan_policy_override_reject_always_fails(tmp_path: Path) -> None:
    csv_path = tmp_path / "mlp_missing_reject.csv"
    df = pd.DataFrame(
        {
            "num_tokens": [128, 256],
            "time_stats.op.min": [1.0, 2.0],
            "time_stats.op.max": [1.0, 2.0],
            "time_stats.op.mean": [1.0, 2.0],
            "time_stats.op.median": [None, 2.0],
        }
    )
    df.to_csv(csv_path, index=False)

    with pytest.raises(MlpCsvValidationError):
        validate_mlp_csv(csv_path, mode="non_strict", nan_policy="reject")


def test_validate_mlp_csv_nan_policy_override_drop_allows_in_strict(tmp_path: Path) -> None:
    csv_path = tmp_path / "mlp_missing_drop.csv"
    df = pd.DataFrame(
        {
            "num_tokens": [128, 256],
            "time_stats.op.min": [1.0, 2.0],
            "time_stats.op.max": [1.0, 2.0],
            "time_stats.op.mean": [1.0, 2.0],
            "time_stats.op.median": [None, 2.0],
        }
    )
    df.to_csv(csv_path, index=False)

    result = validate_mlp_csv(csv_path, mode="strict", nan_policy="drop")
    assert result.missing_cells_total > 0
    assert result.nan_policy_effective == "drop"
    assert result.warnings


def test_validate_mlp_csv_zero_heavy_strict_fails_non_strict_warns(tmp_path: Path) -> None:
    csv_path = tmp_path / "mlp_zero_heavy.csv"
    df = pd.DataFrame(
        {
            "num_tokens": [256, 512],
            "time_stats.op.min": [0.0, 0.0],
            "time_stats.op.max": [0.0, 0.0],
            "time_stats.op.mean": [0.0, 0.0],
            "time_stats.op.median": [0.0, 0.0],
        }
    )
    df.to_csv(csv_path, index=False)

    with pytest.raises(MlpCsvValidationError) as excinfo:
        validate_mlp_csv(csv_path, mode="strict", small_input_threshold=128, zero_heavy_limit=0.01)
    assert excinfo.value.result.zero_heavy_columns

    result = validate_mlp_csv(csv_path, mode="non_strict", small_input_threshold=128, zero_heavy_limit=0.01)
    assert result.zero_heavy_columns
    assert result.warnings


def test_validate_mlp_csv_zeros_below_threshold_ok(tmp_path: Path) -> None:
    csv_path = tmp_path / "mlp_threshold_ok.csv"
    df = pd.DataFrame(
        {
            "num_tokens": [64, 96, 128, 256],
            "time_stats.op.min": [0.0, 0.0, 1.0, 2.0],
            "time_stats.op.max": [0.0, 0.0, 1.0, 2.0],
            "time_stats.op.mean": [0.0, 0.0, 1.0, 2.0],
            "time_stats.op.median": [0.0, 0.0, 1.0, 2.0],
        }
    )
    df.to_csv(csv_path, index=False)

    result = validate_mlp_csv(csv_path, mode="strict", small_input_threshold=128, zero_heavy_limit=0.01)
    assert result.zero_heavy_columns == []
    assert result.warnings == []


def test_validate_mlp_csv_missing_core_columns_fail(tmp_path: Path) -> None:
    csv_path = tmp_path / "mlp_missing_cols.csv"
    df = pd.DataFrame({"num_tokens": [128], "time_stats.op.median": [1.0]})
    df.to_csv(csv_path, index=False)

    with pytest.raises(MlpCsvValidationError) as excinfo:
        validate_mlp_csv(csv_path, mode="strict")

    assert "time_stats.op.min" in excinfo.value.result.missing_columns
