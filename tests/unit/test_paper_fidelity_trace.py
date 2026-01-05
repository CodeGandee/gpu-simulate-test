from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gpu_simulate_test.paper_fidelity.traces import (
    TraceSpec,
    add_poisson_arrivals,
    make_static,
    read_trace_csv,
    validate_trace,
)


def test_trace_validation_rejects_negative_tokens() -> None:
    df = pd.DataFrame(
        {
            "arrived_at": [0.0, 1.0],
            "num_prefill_tokens": [1, 1],
            "num_decode_tokens": [1, -1],
        }
    )
    with pytest.raises(ValueError, match="num_decode_tokens"):
        validate_trace(df, spec=TraceSpec(max_tokens=4096))


def test_trace_validation_rejects_unsorted_arrivals() -> None:
    df = pd.DataFrame(
        {
            "arrived_at": [0.0, 2.0, 1.0],
            "num_prefill_tokens": [1, 1, 1],
            "num_decode_tokens": [1, 1, 1],
        }
    )
    with pytest.raises(ValueError, match="non-decreasing"):
        validate_trace(df, spec=TraceSpec(max_tokens=4096))


def test_static_trace_sets_all_arrivals_to_zero() -> None:
    df = pd.DataFrame(
        {
            "arrived_at": [0.1, 0.2],
            "num_prefill_tokens": [10, 11],
            "num_decode_tokens": [1, 2],
        }
    )
    out = make_static(df)
    assert out["arrived_at"].tolist() == [0.0, 0.0]


def test_poisson_arrivals_are_deterministic_for_seed() -> None:
    base = pd.DataFrame(
        {
            "arrived_at": [0.0, 0.0, 0.0, 0.0],
            "num_prefill_tokens": [1, 1, 1, 1],
            "num_decode_tokens": [1, 1, 1, 1],
        }
    )
    a = add_poisson_arrivals(base, qps=2.0, seed=123)
    b = add_poisson_arrivals(base, qps=2.0, seed=123)
    assert a["arrived_at"].tolist() == b["arrived_at"].tolist()
    assert all(t >= 0.0 for t in a["arrived_at"].tolist())
    assert a["arrived_at"].is_monotonic_increasing


def test_read_trace_csv_validates(tmp_path: Path) -> None:
    path = tmp_path / "trace.csv"
    df = pd.DataFrame(
        {
            "arrived_at": [0.0],
            "num_prefill_tokens": [1],
            "num_decode_tokens": [1],
        }
    )
    df.to_csv(path, index=False)
    out = read_trace_csv(path, spec=TraceSpec(max_tokens=4096))
    assert list(out.columns) == ["arrived_at", "num_prefill_tokens", "num_decode_tokens"]

