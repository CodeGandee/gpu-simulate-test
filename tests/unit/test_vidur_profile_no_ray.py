"""
Unit tests for "no-Ray compute profiling" gating.

These tests validate that unsupported no-Ray configurations fail fast before Ray-dependent
profiling entrypoints are invoked.
"""

from __future__ import annotations

import pytest

from gpu_simulate_test.vidur_cli.errors import UserFacingError
from gpu_simulate_test.vidur_cli.stages import _validate_no_ray_compute_profiling


def test_no_ray_is_not_supported() -> None:
    with pytest.raises(UserFacingError, match="profiling\\.compute\\.use_ray=false"):
        _validate_no_ray_compute_profiling(include_cpu_overhead=False)


def test_no_ray_error_mentions_vidur_stub() -> None:
    with pytest.raises(UserFacingError) as exc:
        _validate_no_ray_compute_profiling(include_cpu_overhead=True)
    assert "disable_ray" in (exc.value.hint or "")
