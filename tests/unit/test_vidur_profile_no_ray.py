"""
Unit tests for "no-Ray compute profiling" gating.

These tests validate that unsupported no-Ray configurations fail fast before Ray-dependent
profiling entrypoints are invoked.
"""

from __future__ import annotations

import pytest

from gpu_simulate_test.vidur_cli.errors import UserFacingError
from gpu_simulate_test.vidur_cli.stages import _validate_no_ray_compute_profiling


def test_no_ray_rejects_multi_gpu() -> None:
    with pytest.raises(UserFacingError, match="single-GPU"):
        _validate_no_ray_compute_profiling(num_gpus=2, tensor_parallel_size=1, include_cpu_overhead=False)


def test_no_ray_rejects_tensor_parallel() -> None:
    with pytest.raises(UserFacingError, match="tensor parallel"):
        _validate_no_ray_compute_profiling(num_gpus=1, tensor_parallel_size=2, include_cpu_overhead=False)


def test_no_ray_rejects_cpu_overhead() -> None:
    with pytest.raises(UserFacingError, match="cpu overhead"):
        _validate_no_ray_compute_profiling(num_gpus=1, tensor_parallel_size=1, include_cpu_overhead=True)

