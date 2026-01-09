"""
Unit tests for Vidur profiling-root validation.

These tests exercise `gpu_simulate_test.vidur_ext.profiling_root.validate_profiling_root` with
minimal dummy CSV files to ensure required files are enforced correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_simulate_test.vidur_ext.profiling_root import ProfilingRootLayout, validate_profiling_root


def _write_dummy_csv(path: Path) -> None:
    """Write a one-column CSV file at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("col\n", encoding="utf-8")


def test_validate_profiling_root_tp1_pp1(tmp_path: Path) -> None:
    model_id = "meta-llama/Llama-2-7b-hf"
    device = "a100"
    network_device = "a100_pairwise_nvlink"

    _write_dummy_csv(tmp_path / "data" / "profiling" / "compute" / device / model_id / "mlp.csv")
    _write_dummy_csv(tmp_path / "data" / "profiling" / "compute" / device / model_id / "attention.csv")

    layout = ProfilingRootLayout(
        profiling_root=tmp_path,
        device=device,
        model_id=model_id,
        network_device=network_device,
        tensor_parallel_size=1,
        num_pipeline_stages=1,
        enable_cpu_overhead_modeling=False,
    )
    validate_profiling_root(layout)


def test_validate_profiling_root_requires_network_for_tp_gt_1(tmp_path: Path) -> None:
    model_id = "meta-llama/Llama-2-7b-hf"
    device = "a100"
    network_device = "a100_pairwise_nvlink"

    _write_dummy_csv(tmp_path / "data" / "profiling" / "compute" / device / model_id / "mlp.csv")
    _write_dummy_csv(tmp_path / "data" / "profiling" / "compute" / device / model_id / "attention.csv")

    layout = ProfilingRootLayout(
        profiling_root=tmp_path,
        device=device,
        model_id=model_id,
        network_device=network_device,
        tensor_parallel_size=2,
        num_pipeline_stages=1,
        enable_cpu_overhead_modeling=False,
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        validate_profiling_root(layout)
    assert "all_reduce.csv" in str(excinfo.value)


def test_validate_profiling_root_requires_cpu_overheads_when_enabled(tmp_path: Path) -> None:
    model_id = "meta-llama/Llama-2-7b-hf"
    device = "a100"
    network_device = "a100_pairwise_nvlink"

    _write_dummy_csv(tmp_path / "data" / "profiling" / "compute" / device / model_id / "mlp.csv")
    _write_dummy_csv(tmp_path / "data" / "profiling" / "compute" / device / model_id / "attention.csv")

    layout = ProfilingRootLayout(
        profiling_root=tmp_path,
        device=device,
        model_id=model_id,
        network_device=network_device,
        tensor_parallel_size=1,
        num_pipeline_stages=1,
        enable_cpu_overhead_modeling=True,
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        validate_profiling_root(layout)
    assert "cpu_overheads.csv" in str(excinfo.value)
