"""
Trace IO, validation, and deterministic conversions for paper-fidelity workflows.

The canonical trace format is a single CSV (`trace.csv`) that both Vidur simulation and
Sarathi-Serve real replay can consume.

Schema (`trace.csv`)
--------------------
Required columns
    - arrived_at: float (seconds since start; non-decreasing; >= 0)
    - num_prefill_tokens: int (>= 1)
    - num_decode_tokens: int (>= 1)
Optional columns
    - request_id: int
    - prompt_id: str
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from gpu_simulate_test.io import read_csv, write_csv


TRACE_REQUIRED_COLUMNS = ("arrived_at", "num_prefill_tokens", "num_decode_tokens")


@dataclass(frozen=True)
class TraceSpec:
    """Parameters for generating/validating a canonical trace."""

    max_tokens: int = 4096
    seed: int = 42
    num_requests: int | None = None


TraceSubsetKind = Literal["all", "range", "indices"]


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"trace_subset.{name} must be an int (got bool)")
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise ValueError(f"trace_subset.{name} must be an int (got {value!r})")


def _require_int_list(name: str, values: Sequence[object]) -> list[int]:
    out: list[int] = []
    for v in values:
        out.append(_require_int(name, v))
    return out


def apply_trace_subset(
    df: pd.DataFrame,
    *,
    spec: TraceSpec,
    kind: TraceSubsetKind | str,
    begin: int | None = None,
    end: int | None = None,
    indices: Sequence[object] | None = None,
    allow_indices: bool,
    rebase_arrived_at: bool,
) -> pd.DataFrame:
    """Return a subset of trace rows, validating inputs and preserving row order.

    Parameters
    ----------
    df:
        Canonical trace dataframe.
    spec:
        Trace validation parameters.
    kind:
        Subset mode: "all", "range", or "indices".
    begin, end:
        Row bounds for "range" selection (Python slicing semantics, end-exclusive). `None` means
        open-ended (begin defaults to 0; end defaults to len(df)).
    indices:
        Discrete row indices for "indices" selection. Only allowed when `allow_indices=True`.
    allow_indices:
        If False, `kind="indices"` is rejected with an actionable error. This is intended for
        already-timed trace sources where non-contiguous selection is ambiguous.
    rebase_arrived_at:
        If True and `kind="range"`, subtract the first selected row's `arrived_at` so the subset
        starts at time 0.0.
    """
    kind_str = str(kind)
    if kind_str == "all":
        out = df.copy()
        validate_trace(out, spec=spec)
        return out

    n = len(df)
    if n == 0:
        raise ValueError("trace_subset: cannot select from an empty trace")

    if kind_str == "range":
        b = 0 if begin is None else _require_int("begin", begin)
        e = n if end is None else _require_int("end", end)

        if b < 0:
            raise ValueError(f"trace_subset.begin must be >= 0 (got {b})")
        if e < 0:
            raise ValueError(f"trace_subset.end must be >= 0 (got {e})")
        if b > n:
            raise ValueError(f"trace_subset.begin out of bounds: begin={b} len={n}")
        if e > n:
            raise ValueError(f"trace_subset.end out of bounds: end={e} len={n}")
        if b >= e:
            raise ValueError(f"trace_subset range must be non-empty (got begin={b}, end={e})")

        out = df.iloc[b:e].copy().reset_index(drop=True)
        if rebase_arrived_at:
            t0 = float(pd.to_numeric(out["arrived_at"], errors="raise").iloc[0])
            out["arrived_at"] = pd.to_numeric(out["arrived_at"], errors="raise").astype(float) - t0

        validate_trace(out, spec=spec)
        return out

    if kind_str == "indices":
        if not allow_indices:
            raise ValueError(
                "trace_subset.kind=indices is only supported for untimed trace sources (those where arrivals are "
                "generated inside the workflow); for timed trace sources, use trace_subset.kind=range instead."
            )
        if indices is None:
            raise ValueError("trace_subset.kind=indices requires trace_subset.indices")

        idx = _require_int_list("indices", indices)
        if len(idx) == 0:
            raise ValueError("trace_subset.indices must be non-empty")

        seen: set[int] = set()
        dup: set[int] = set()
        for i in idx:
            if i in seen:
                dup.add(i)
            seen.add(i)
        if dup:
            raise ValueError(f"trace_subset.indices contains duplicate indices (e.g. {sorted(dup)[:5]})")

        lo = min(idx)
        hi = max(idx)
        if lo < 0 or hi >= n:
            raise ValueError(f"trace_subset.indices out of bounds for len={n}: min={lo} max={hi}")

        out = df.iloc[sorted(idx)].copy().reset_index(drop=True)
        validate_trace(out, spec=spec)
        return out

    raise ValueError(f"trace_subset.kind must be one of ['all', 'range', 'indices'] (got {kind_str!r})")


def read_trace_csv(path: Path, *, spec: TraceSpec) -> pd.DataFrame:
    """Load and validate a canonical `trace.csv`.

    Parameters
    ----------
    path:
        Path to the trace CSV.
    spec:
        Validation/generation parameters (max token budget and sampling controls).

    Returns
    -------
    pandas.DataFrame
        The loaded trace dataframe.
    """
    df = pd.read_csv(path)
    validate_trace(df, spec=spec)
    return df


def validate_trace(df: pd.DataFrame, *, spec: TraceSpec) -> None:
    """Fail fast with actionable errors if schema/values are invalid."""
    missing = [c for c in TRACE_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"trace.csv: missing required columns: {missing}")

    arrived_at = pd.to_numeric(df["arrived_at"], errors="raise").astype(float)
    if arrived_at.isna().any() or not np.isfinite(arrived_at.to_numpy()).all():
        raise ValueError("trace.csv: arrived_at must be finite (no NaN/inf)")
    if (arrived_at < 0).any():
        raise ValueError("trace.csv: arrived_at must be >= 0")
    if not arrived_at.is_monotonic_increasing:
        raise ValueError("trace.csv: arrived_at must be non-decreasing")

    num_prefill_tokens = pd.to_numeric(df["num_prefill_tokens"], errors="raise")
    num_decode_tokens = pd.to_numeric(df["num_decode_tokens"], errors="raise")

    for name, series in [
        ("num_prefill_tokens", num_prefill_tokens),
        ("num_decode_tokens", num_decode_tokens),
    ]:
        if series.isna().any() or not np.isfinite(series.to_numpy()).all():
            raise ValueError(f"trace.csv: {name} must be finite (no NaN/inf)")
        if (series < 1).any():
            raise ValueError(f"trace.csv: {name} must be >= 1")
        if not (series.astype(int) == series).all():
            raise ValueError(f"trace.csv: {name} must be integer-valued")

    total_tokens = num_prefill_tokens.astype(int) + num_decode_tokens.astype(int)
    if (total_tokens > int(spec.max_tokens)).any():
        raise ValueError(f"trace.csv: (num_prefill_tokens + num_decode_tokens) must be <= {spec.max_tokens}")


def legacy_workload_dir_to_trace(workload_dir: Path, *, out_csv: Path, spec: TraceSpec) -> pd.DataFrame:
    """Convert legacy split files into canonical `trace.csv` deterministically.

    Uses:
    - `trace_lengths.csv`: request_id,prompt_id,num_prefill_tokens,num_decode_tokens
    - `trace_intervals.csv`: request_id,inter_arrival_ns,arrival_time_ns

    Canonicalization:
    - `arrived_at = arrival_time_ns / 1e9`
    """
    lengths = read_csv(
        workload_dir / "trace_lengths.csv",
        required_columns=["request_id", "prompt_id", "num_prefill_tokens", "num_decode_tokens"],
        context="trace_lengths",
    )
    intervals = read_csv(
        workload_dir / "trace_intervals.csv",
        required_columns=["request_id", "inter_arrival_ns", "arrival_time_ns"],
        context="trace_intervals",
    )

    merged = pd.merge(lengths, intervals, on=["request_id"], how="inner").sort_values("request_id")
    merged = merged.reset_index(drop=True)

    df = pd.DataFrame(
        {
            "arrived_at": merged["arrival_time_ns"].astype(float) / 1e9,
            "num_prefill_tokens": merged["num_prefill_tokens"].astype(int),
            "num_decode_tokens": merged["num_decode_tokens"].astype(int),
            "request_id": merged["request_id"].astype(int),
            "prompt_id": merged["prompt_id"].astype(str),
        }
    )
    validate_trace(df, spec=spec)
    write_csv(out_csv, df, required_columns=list(TRACE_REQUIRED_COLUMNS))
    return df


def processed_lengths_csv_to_trace(lengths_csv: Path, *, spec: TraceSpec) -> pd.DataFrame:
    """Load a token-length distribution CSV and return a canonical trace frame (arrivals unset).

    The returned dataframe has `arrived_at` initialized to 0.0 for all rows; callers should apply
    `make_static` or `add_poisson_arrivals` to set arrivals as needed.
    """
    df = pd.read_csv(lengths_csv)
    missing = [c for c in ["num_prefill_tokens", "num_decode_tokens"] if c not in df.columns]
    if missing:
        raise ValueError(f"{lengths_csv}: missing required columns: {missing}")

    df = df[["num_prefill_tokens", "num_decode_tokens"]].copy()
    df["num_prefill_tokens"] = pd.to_numeric(df["num_prefill_tokens"], errors="raise").astype(int)
    df["num_decode_tokens"] = pd.to_numeric(df["num_decode_tokens"], errors="raise").astype(int)

    total = df["num_prefill_tokens"] + df["num_decode_tokens"]
    df = df[total <= int(spec.max_tokens)].reset_index(drop=True)

    if spec.num_requests is not None:
        n = int(spec.num_requests)
        if n <= 0:
            raise ValueError("num_requests must be >= 1 when provided")
        if n > len(df):
            raise ValueError(f"num_requests={n} exceeds available rows ({len(df)})")
        rng = np.random.default_rng(int(spec.seed))
        idx = np.arange(len(df))
        rng.shuffle(idx)
        df = df.iloc[idx[:n]].reset_index(drop=True)

    out = pd.DataFrame(
        {
            "arrived_at": np.zeros(len(df), dtype=float),
            "num_prefill_tokens": df["num_prefill_tokens"].astype(int),
            "num_decode_tokens": df["num_decode_tokens"].astype(int),
            "request_id": np.arange(len(df), dtype=int),
        }
    )
    validate_trace(out, spec=spec)
    return out


def make_static(df: pd.DataFrame) -> pd.DataFrame:
    """Return a static trace with all arrivals at time 0."""
    out = df.copy()
    out["arrived_at"] = 0.0
    return out


def add_poisson_arrivals(df: pd.DataFrame, *, qps: float, seed: int) -> pd.DataFrame:
    """Assign arrival times using a seeded Poisson arrival process."""
    if qps <= 0:
        raise ValueError(f"qps must be > 0 (got {qps})")

    out = df.copy()
    n = len(out)
    if n == 0:
        return out

    rng = np.random.default_rng(int(seed))
    if n == 1:
        arrivals = np.array([0.0], dtype=float)
    else:
        inter_arrivals = rng.exponential(scale=1.0 / float(qps), size=n - 1)
        arrivals = np.concatenate([np.array([0.0], dtype=float), np.cumsum(inter_arrivals).astype(float)])

    out["arrived_at"] = arrivals
    return out
