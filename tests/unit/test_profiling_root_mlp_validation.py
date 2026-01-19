from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gpu_simulate_test.vidur_ext.mlp_validation import MlpCsvValidationError
from gpu_simulate_test.vidur_ext.profiling_root import ProfilingRootLayout, validate_profiling_root


def _write_minimal_root(*, root: Path, model_id: str, mlp_df: pd.DataFrame) -> None:
    compute_dir = root / "data" / "profiling" / "compute" / "a100" / model_id
    compute_dir.mkdir(parents=True, exist_ok=True)
    (compute_dir / "attention.csv").write_text("x\n1\n", encoding="utf-8")
    mlp_df.to_csv(compute_dir / "mlp.csv", index=False)


def test_validate_profiling_root_strict_fails_on_zero_heavy(tmp_path: Path) -> None:
    model_id = "test/model"
    df = pd.DataFrame(
        {
            "num_tokens": [256, 512],
            "time_stats.op.min": [0.0, 0.0],
            "time_stats.op.max": [0.0, 0.0],
            "time_stats.op.mean": [0.0, 0.0],
            "time_stats.op.median": [0.0, 0.0],
        }
    )
    _write_minimal_root(root=tmp_path, model_id=model_id, mlp_df=df)

    layout = ProfilingRootLayout(profiling_root=tmp_path, device="a100", model_id=model_id)
    with pytest.raises(MlpCsvValidationError):
        validate_profiling_root(layout)


def test_validate_profiling_root_non_strict_warns_on_zero_heavy(tmp_path: Path) -> None:
    model_id = "test/model"
    df = pd.DataFrame(
        {
            "num_tokens": [256, 512],
            "time_stats.op.min": [0.0, 0.0],
            "time_stats.op.max": [0.0, 0.0],
            "time_stats.op.mean": [0.0, 0.0],
            "time_stats.op.median": [0.0, 0.0],
        }
    )
    _write_minimal_root(root=tmp_path, model_id=model_id, mlp_df=df)

    layout = ProfilingRootLayout(
        profiling_root=tmp_path,
        device="a100",
        model_id=model_id,
        mlp_validation_mode="non_strict",
    )
    with pytest.warns(UserWarning):
        validate_profiling_root(layout)


def test_validate_profiling_root_missing_values_auto_non_strict_warns(tmp_path: Path) -> None:
    model_id = "test/model"
    df = pd.DataFrame(
        {
            "num_tokens": [256, 512],
            "time_stats.op.min": [1.0, 2.0],
            "time_stats.op.max": [1.0, 2.0],
            "time_stats.op.mean": [1.0, None],
            "time_stats.op.median": [1.0, 2.0],
        }
    )
    _write_minimal_root(root=tmp_path, model_id=model_id, mlp_df=df)

    layout = ProfilingRootLayout(
        profiling_root=tmp_path,
        device="a100",
        model_id=model_id,
        mlp_validation_mode="non_strict",
    )
    with pytest.warns(UserWarning):
        validate_profiling_root(layout)


def test_validate_profiling_root_missing_values_reject_override_fails(tmp_path: Path) -> None:
    model_id = "test/model"
    df = pd.DataFrame(
        {
            "num_tokens": [256, 512],
            "time_stats.op.min": [1.0, 2.0],
            "time_stats.op.max": [1.0, 2.0],
            "time_stats.op.mean": [1.0, None],
            "time_stats.op.median": [1.0, 2.0],
        }
    )
    _write_minimal_root(root=tmp_path, model_id=model_id, mlp_df=df)

    layout = ProfilingRootLayout(
        profiling_root=tmp_path,
        device="a100",
        model_id=model_id,
        mlp_validation_mode="non_strict",
        mlp_nan_policy="reject",
    )
    with pytest.raises(MlpCsvValidationError):
        validate_profiling_root(layout)


def test_validate_profiling_root_missing_values_drop_override_allows_in_strict(tmp_path: Path) -> None:
    model_id = "test/model"
    df = pd.DataFrame(
        {
            "num_tokens": [256, 512],
            "time_stats.op.min": [1.0, 2.0],
            "time_stats.op.max": [1.0, 2.0],
            "time_stats.op.mean": [1.0, None],
            "time_stats.op.median": [1.0, 2.0],
        }
    )
    _write_minimal_root(root=tmp_path, model_id=model_id, mlp_df=df)

    layout = ProfilingRootLayout(
        profiling_root=tmp_path,
        device="a100",
        model_id=model_id,
        mlp_validation_mode="strict",
        mlp_nan_policy="drop",
    )
    with pytest.warns(UserWarning):
        validate_profiling_root(layout)


def test_validate_profiling_root_returns_mlp_result_when_cpu_overhead_enabled(tmp_path: Path) -> None:
    model_id = "test/model"
    df = pd.DataFrame(
        {
            "num_tokens": [256, 512],
            "time_stats.op.min": [1.0, 2.0],
            "time_stats.op.max": [1.0, 2.0],
            "time_stats.op.mean": [1.0, 2.0],
            "time_stats.op.median": [1.0, 2.0],
        }
    )
    _write_minimal_root(root=tmp_path, model_id=model_id, mlp_df=df)

    cpu_dir = tmp_path / "data" / "profiling" / "cpu_overhead" / "a100_pairwise_nvlink" / model_id
    cpu_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "model_name": [model_id],
            "batch_size": [1],
            "tensor_parallel_degree": [1],
        }
    ).to_csv(cpu_dir / "cpu_overheads.csv", index=False)

    layout = ProfilingRootLayout(
        profiling_root=tmp_path,
        device="a100",
        model_id=model_id,
        skip_cpu_overhead_modeling=False,
    )
    result = validate_profiling_root(layout)
    assert result.csv_path.name == "mlp.csv"
