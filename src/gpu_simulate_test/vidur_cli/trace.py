"""
Canonical trace generation for `vidur-cli`.

This module creates and validates the canonical token-length trace files under
`<run_dir>/trace/`.

The canonical schema is a CSV with required columns:
- `request_id` (int, unique)
- `arrival_time_ns` (int, >= 0, non-decreasing)
- `num_prefill_tokens` (int, >= 1)
- `num_decode_tokens` (int, >= 1)

The module also produces compatibility artifacts used by legacy runners:
- `trace_lengths.csv`
- `trace_intervals.csv`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from gpu_simulate_test.io import assert_columns, utcnow_iso, write_csv, write_json
from gpu_simulate_test.workloads.arrival_schedule import ArrivalScheduleConfig, build_trace_intervals
from gpu_simulate_test.workloads.prompts import read_prompts_jsonl
from gpu_simulate_test.workloads.token_lengths import (
    TraceLengthsConfig,
    compute_trace_lengths,
    load_hf_tokenizer,
)


TRACE_REQUIRED_COLUMNS = ["request_id", "arrival_time_ns", "num_prefill_tokens", "num_decode_tokens"]


@dataclass(frozen=True)
class TraceBuildResult:
    trace_csv: Path
    trace_meta_json: Path
    trace_lengths_csv: Path
    trace_intervals_csv: Path


def validate_canonical_trace(df: pd.DataFrame) -> None:
    """Validate canonical trace invariants (raises ValueError on failures)."""
    assert_columns(df, TRACE_REQUIRED_COLUMNS, context="trace.csv")

    request_id = pd.to_numeric(df["request_id"], errors="raise")
    arrival_time_ns = pd.to_numeric(df["arrival_time_ns"], errors="raise")
    num_prefill_tokens = pd.to_numeric(df["num_prefill_tokens"], errors="raise")
    num_decode_tokens = pd.to_numeric(df["num_decode_tokens"], errors="raise")

    if request_id.isna().any():
        raise ValueError("trace.csv: request_id contains NaN values")
    if arrival_time_ns.isna().any():
        raise ValueError("trace.csv: arrival_time_ns contains NaN values")
    if num_prefill_tokens.isna().any():
        raise ValueError("trace.csv: num_prefill_tokens contains NaN values")
    if num_decode_tokens.isna().any():
        raise ValueError("trace.csv: num_decode_tokens contains NaN values")

    request_id_i64 = request_id.astype("int64")
    arrival_i64 = arrival_time_ns.astype("int64")
    prefill_i64 = num_prefill_tokens.astype("int64")
    decode_i64 = num_decode_tokens.astype("int64")

    if request_id_i64.duplicated().any():
        dup = request_id_i64.loc[request_id_i64.duplicated()].unique()[:5].tolist()
        raise ValueError(f"trace.csv: request_id must be unique; duplicates: {dup}")

    if (arrival_i64 < 0).any():
        bad = arrival_i64.loc[arrival_i64 < 0].head(5).tolist()
        raise ValueError(f"trace.csv: arrival_time_ns must be >= 0; got examples {bad}")
    diffs = arrival_i64.diff().fillna(0).astype("int64")
    if (diffs < 0).any():
        idx = diffs.loc[diffs < 0].index[:5].tolist()
        raise ValueError(f"trace.csv: arrival_time_ns must be non-decreasing; decreases at rows {idx}")

    if (prefill_i64 < 1).any():
        bad = prefill_i64.loc[prefill_i64 < 1].head(5).tolist()
        raise ValueError(f"trace.csv: num_prefill_tokens must be >= 1; got examples {bad}")
    if (decode_i64 < 1).any():
        bad = decode_i64.loc[decode_i64 < 1].head(5).tolist()
        raise ValueError(f"trace.csv: num_decode_tokens must be >= 1; got examples {bad}")


def import_canonical_trace(*, src_csv: Path, out_dir: Path) -> TraceBuildResult:
    """Validate and copy an existing canonical trace CSV into a run directory."""
    src_csv = src_csv.expanduser().resolve()
    if not src_csv.exists():
        raise FileNotFoundError(f"import trace CSV does not exist: {src_csv}")

    df = pd.read_csv(src_csv)
    validate_canonical_trace(df)

    # Ensure request_id is sequential for compatibility with legacy runners.
    original_request_id = df["request_id"].copy()
    df["request_id"] = pd.Series(range(len(df)), dtype="int64")
    if not original_request_id.astype("int64").equals(df["request_id"]):
        df["import_request_id"] = original_request_id

    trace_dir = out_dir.expanduser().resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)

    trace_csv = trace_dir / "trace.csv"
    df.to_csv(trace_csv, index=False)

    trace_lengths = trace_dir / "trace_lengths.csv"
    trace_intervals = trace_dir / "trace_intervals.csv"
    _write_compatibility_artifacts(trace= df, trace_dir=trace_dir, trace_lengths=trace_lengths, trace_intervals=trace_intervals)

    trace_meta = trace_dir / "trace_meta.json"
    meta = _trace_meta_payload(
        trace_csv=trace_csv,
        source={"kind": "import", "path": str(src_csv)},
        arrival_schedule={"kind": "fixed_interval", "seed": 0, "note": "imported"},
    )
    write_json(trace_meta, meta)
    return TraceBuildResult(
        trace_csv=trace_csv.resolve(),
        trace_meta_json=trace_meta.resolve(),
        trace_lengths_csv=trace_lengths.resolve(),
        trace_intervals_csv=trace_intervals.resolve(),
    )


def build_from_lengths_csv(
    *,
    lengths_csv: Path,
    schedule: ArrivalScheduleConfig,
    out_dir: Path,
) -> TraceBuildResult:
    """Create canonical trace from a lengths-only CSV deterministically."""
    lengths_csv = lengths_csv.expanduser().resolve()
    if not lengths_csv.exists():
        raise FileNotFoundError(f"lengths CSV does not exist: {lengths_csv}")

    lengths = pd.read_csv(lengths_csv)
    assert_columns(lengths, ["num_prefill_tokens", "num_decode_tokens"], context=str(lengths_csv))
    n = int(len(lengths))

    intervals = build_trace_intervals(n, config=schedule)
    trace = pd.DataFrame(
        {
            "request_id": intervals["request_id"].astype("int64"),
            "arrival_time_ns": intervals["arrival_time_ns"].astype("int64"),
            "num_prefill_tokens": pd.to_numeric(lengths["num_prefill_tokens"], errors="raise").astype("int64"),
            "num_decode_tokens": pd.to_numeric(lengths["num_decode_tokens"], errors="raise").astype("int64"),
        }
    )
    validate_canonical_trace(trace)

    trace_dir = out_dir.expanduser().resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)

    trace_csv = trace_dir / "trace.csv"
    write_csv(trace_csv, trace, required_columns=TRACE_REQUIRED_COLUMNS)

    trace_lengths = trace_dir / "trace_lengths.csv"
    trace_intervals = trace_dir / "trace_intervals.csv"
    _write_compatibility_artifacts(trace=trace, trace_dir=trace_dir, trace_lengths=trace_lengths, trace_intervals=trace_intervals)

    trace_meta = trace_dir / "trace_meta.json"
    meta = _trace_meta_payload(
        trace_csv=trace_csv,
        source={"kind": "lengths_csv", "path": str(lengths_csv)},
        arrival_schedule=_arrival_schedule_payload(schedule),
    )
    write_json(trace_meta, meta)

    return TraceBuildResult(
        trace_csv=trace_csv.resolve(),
        trace_meta_json=trace_meta.resolve(),
        trace_lengths_csv=trace_lengths.resolve(),
        trace_intervals_csv=trace_intervals.resolve(),
    )


def build_default_trace(
    *,
    prompts_jsonl: Path,
    tokenizer_ref: Path,
    num_decode_tokens: int,
    schedule: ArrivalScheduleConfig,
    out_dir: Path,
) -> TraceBuildResult:
    """Generate a canonical trace from prompts + tokenizer + arrival schedule."""
    prompts_jsonl = prompts_jsonl.expanduser().resolve()
    tokenizer_ref = tokenizer_ref.expanduser().resolve()
    if not prompts_jsonl.exists():
        raise FileNotFoundError(f"prompts file does not exist: {prompts_jsonl}")

    prompts = read_prompts_jsonl(prompts_jsonl)
    tokenizer = load_hf_tokenizer(tokenizer_ref)

    lengths = compute_trace_lengths(
        prompts,
        tokenizer=tokenizer,
        config=TraceLengthsConfig(num_decode_tokens=int(num_decode_tokens)),
    )
    assert_columns(lengths, ["request_id", "prompt_id", "num_prefill_tokens", "num_decode_tokens"], context="lengths")

    intervals = build_trace_intervals(len(prompts), config=schedule)
    merged = pd.merge(lengths, intervals, on=["request_id"], how="inner").sort_values("request_id").reset_index(drop=True)

    trace = merged[["request_id", "arrival_time_ns", "num_prefill_tokens", "num_decode_tokens", "prompt_id"]].copy()
    trace["request_id"] = trace["request_id"].astype("int64")
    trace["arrival_time_ns"] = trace["arrival_time_ns"].astype("int64")
    trace["num_prefill_tokens"] = trace["num_prefill_tokens"].astype("int64")
    trace["num_decode_tokens"] = trace["num_decode_tokens"].astype("int64")
    validate_canonical_trace(trace)

    trace_dir = out_dir.expanduser().resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)

    trace_csv = trace_dir / "trace.csv"
    trace.to_csv(trace_csv, index=False)

    trace_lengths = trace_dir / "trace_lengths.csv"
    trace_intervals = trace_dir / "trace_intervals.csv"
    # For the default path, keep prompt_id from the prompts corpus.
    write_csv(
        trace_lengths,
        lengths[["request_id", "prompt_id", "num_prefill_tokens", "num_decode_tokens"]],
        required_columns=["request_id", "prompt_id", "num_prefill_tokens", "num_decode_tokens"],
    )
    write_csv(trace_intervals, intervals, required_columns=["request_id", "inter_arrival_ns", "arrival_time_ns"])

    trace_meta = trace_dir / "trace_meta.json"
    meta = _trace_meta_payload(
        trace_csv=trace_csv,
        source={
            "kind": "lengths_csv",
            "path": str(trace_lengths.resolve()),
            "generated_from_prompts_jsonl": str(prompts_jsonl),
            "tokenizer_ref": str(tokenizer_ref),
        },
        arrival_schedule=_arrival_schedule_payload(schedule),
    )
    write_json(trace_meta, meta)

    return TraceBuildResult(
        trace_csv=trace_csv.resolve(),
        trace_meta_json=trace_meta.resolve(),
        trace_lengths_csv=trace_lengths.resolve(),
        trace_intervals_csv=trace_intervals.resolve(),
    )


def _write_compatibility_artifacts(
    *,
    trace: pd.DataFrame,
    trace_dir: Path,
    trace_lengths: Path,
    trace_intervals: Path,
) -> None:
    """Write `trace_lengths.csv` and `trace_intervals.csv` from a canonical trace."""
    _ = trace_dir  # explicit for readability
    df = trace.copy()
    validate_canonical_trace(df)

    request_id = pd.to_numeric(df["request_id"], errors="raise").astype("int64")
    arrival_time_ns = pd.to_numeric(df["arrival_time_ns"], errors="raise").astype("int64")
    inter_arrival_ns = arrival_time_ns.diff().fillna(0).astype("int64")

    lengths_df = pd.DataFrame(
        {
            "request_id": request_id,
            "prompt_id": request_id.astype(str),
            "num_prefill_tokens": pd.to_numeric(df["num_prefill_tokens"], errors="raise").astype("int64"),
            "num_decode_tokens": pd.to_numeric(df["num_decode_tokens"], errors="raise").astype("int64"),
        }
    )
    intervals_df = pd.DataFrame(
        {
            "request_id": request_id,
            "inter_arrival_ns": inter_arrival_ns,
            "arrival_time_ns": arrival_time_ns,
        }
    )
    write_csv(
        trace_lengths,
        lengths_df,
        required_columns=["request_id", "prompt_id", "num_prefill_tokens", "num_decode_tokens"],
    )
    write_csv(trace_intervals, intervals_df, required_columns=["request_id", "inter_arrival_ns", "arrival_time_ns"])


def _arrival_schedule_payload(cfg: ArrivalScheduleConfig) -> Mapping[str, Any]:
    payload: dict[str, Any] = {"kind": cfg.kind, "seed": int(cfg.seed)}
    if cfg.kind == "fixed_interval":
        payload["inter_arrival_ns"] = int(cfg.inter_arrival_ns)
    elif cfg.kind == "poisson":
        payload["poisson_rate_per_s"] = float(cfg.poisson_rate_per_s)
    return payload


def _trace_meta_payload(
    *,
    trace_csv: Path,
    source: Mapping[str, Any],
    arrival_schedule: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "v1",
        "created_at": utcnow_iso(),
        "trace_csv": str(trace_csv.resolve()),
        "source": dict(source),
        "arrival_schedule": dict(arrival_schedule),
    }
