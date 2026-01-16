from __future__ import annotations

import pytest

from gpu_simulate_test.vidur_ext.record_function_tracer_v2 import compute_operation_time_stats


def test_compute_operation_time_stats_cuda_runtime_path() -> None:
    trace = [
        {"cat": "user_annotation", "name": "vidur_mlp_up_proj", "ts": 0, "dur": 1000},
        {"cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 100, "dur": 10, "args": {"correlation": 7}},
        {"cat": "kernel", "name": "someKernel", "ts": 120, "dur": 500, "args": {"correlation": 7}},
    ]

    stats = compute_operation_time_stats(trace)

    assert "mlp_up_proj" in stats
    assert stats["mlp_up_proj"]["median"] == pytest.approx(0.5)
    assert stats["mlp_up_proj"]["mean"] == pytest.approx(0.5)
    assert stats["mlp_up_proj"]["std"] == pytest.approx(0.0)


def test_compute_operation_time_stats_cuda_driver_path() -> None:
    trace = [
        {"cat": "user_annotation", "name": "vidur_mlp_down_proj", "ts": 0, "dur": 1000},
        {"cat": "cuda_driver", "name": "cuLaunchKernel", "ts": 100, "dur": 10, "args": {"correlation": 11}},
        {"cat": "kernel", "name": "someKernel", "ts": 120, "dur": 750, "args": {"correlation": 11}},
    ]

    stats = compute_operation_time_stats(trace)

    assert "mlp_down_proj" in stats
    assert stats["mlp_down_proj"]["median"] == pytest.approx(0.75)


def test_compute_operation_time_stats_dedupes_duplicate_launches() -> None:
    trace = [
        {"cat": "user_annotation", "name": "vidur_attn_pre_proj", "ts": 0, "dur": 1000},
        {"cat": "cuda_driver", "name": "cuLaunchKernel", "ts": 100, "dur": 10, "args": {"correlation": 3}},
        {"cat": "cuda_driver", "name": "cuLaunchKernel", "ts": 110, "dur": 10, "args": {"correlation": 3}},
        {"cat": "kernel", "name": "someKernel", "ts": 120, "dur": 400, "args": {"correlation": 3}},
        {"cat": "kernel", "name": "uncorrelatedKernel", "ts": 130, "dur": 999, "args": {"correlation": 999}},
    ]

    stats = compute_operation_time_stats(trace)

    assert stats["attn_pre_proj"]["mean"] == pytest.approx(0.4)

