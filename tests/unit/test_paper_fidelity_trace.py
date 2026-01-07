from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gpu_simulate_test.paper_fidelity.traces import (
    TraceSpec,
    add_poisson_arrivals,
    apply_trace_subset,
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


def test_trace_subset_range_rebases_arrivals_for_timed_trace() -> None:
    df = pd.DataFrame(
        {
            "arrived_at": [5.0, 5.5, 6.0, 7.0],
            "num_prefill_tokens": [1, 1, 1, 1],
            "num_decode_tokens": [1, 1, 1, 1],
        }
    )
    out = apply_trace_subset(
        df,
        spec=TraceSpec(max_tokens=4096),
        kind="range",
        begin=1,
        end=3,
        indices=None,
        allow_indices=False,
        rebase_arrived_at=True,
    )
    assert out["arrived_at"].tolist() == pytest.approx([0.0, 0.5])


def test_trace_subset_indices_sorts_and_preserves_row_order() -> None:
    df = pd.DataFrame(
        {
            "arrived_at": [0.0, 0.0, 0.0, 0.0],
            "num_prefill_tokens": [10, 11, 12, 13],
            "num_decode_tokens": [1, 1, 1, 1],
            "request_id": [0, 1, 2, 3],
        }
    )
    out = apply_trace_subset(
        df,
        spec=TraceSpec(max_tokens=4096),
        kind="indices",
        begin=None,
        end=None,
        indices=[3, 1],
        allow_indices=True,
        rebase_arrived_at=False,
    )
    assert out["num_prefill_tokens"].tolist() == [11, 13]
    assert out["request_id"].tolist() == [1, 3]


def test_trace_subset_indices_rejects_duplicates() -> None:
    df = pd.DataFrame(
        {
            "arrived_at": [0.0, 0.0],
            "num_prefill_tokens": [1, 1],
            "num_decode_tokens": [1, 1],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        apply_trace_subset(
            df,
            spec=TraceSpec(max_tokens=4096),
            kind="indices",
            indices=[1, 1],
            allow_indices=True,
            rebase_arrived_at=False,
        )


def test_trace_subset_rejects_indices_when_not_allowed() -> None:
    df = pd.DataFrame(
        {
            "arrived_at": [0.0],
            "num_prefill_tokens": [1],
            "num_decode_tokens": [1],
        }
    )
    with pytest.raises(ValueError, match="only supported for untimed"):
        apply_trace_subset(
            df,
            spec=TraceSpec(max_tokens=4096),
            kind="indices",
            indices=[0],
            allow_indices=False,
            rebase_arrived_at=False,
        )
