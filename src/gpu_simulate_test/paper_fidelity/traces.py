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

