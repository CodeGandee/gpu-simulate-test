from __future__ import annotations

from pathlib import Path

import pytest

from gpu_simulate_test.vidur_ext.profiling_root import ProfilingRootLayout, validate_profiling_root


def test_validate_profiling_root_missing_root(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    layout = ProfilingRootLayout(
        profiling_root=missing,
        device="a100",
        model_id="Qwen/Qwen3-0.6B",
    )
    with pytest.raises(FileNotFoundError):
        validate_profiling_root(layout)


def test_validate_profiling_root_missing_files(tmp_path: Path) -> None:
    layout = ProfilingRootLayout(
        profiling_root=tmp_path,
        device="a100",
        model_id="Qwen/Qwen3-0.6B",
    )
    with pytest.raises(FileNotFoundError, match="Missing profiling inputs"):
        validate_profiling_root(layout)


def test_validate_profiling_root_ok(tmp_path: Path) -> None:
    base = tmp_path / "data" / "profiling"
    (base / "compute" / "a100" / "Qwen" / "Qwen3-0.6B").mkdir(parents=True)

    (base / "compute" / "a100" / "Qwen" / "Qwen3-0.6B" / "mlp.csv").write_text("x\n", encoding="utf-8")
    (base / "compute" / "a100" / "Qwen" / "Qwen3-0.6B" / "attention.csv").write_text("x\n", encoding="utf-8")

    layout = ProfilingRootLayout(
        profiling_root=tmp_path,
        device="a100",
        model_id="Qwen/Qwen3-0.6B",
    )
    validate_profiling_root(layout)
