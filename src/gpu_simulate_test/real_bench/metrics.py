from __future__ import annotations

from pathlib import Path

import pandas as pd

from gpu_simulate_test.io import write_csv, write_json
from gpu_simulate_test.real_bench.backends.base import TokenEvent


def build_metrics_frames(
    *,
    request_id: int,
    arrival_time_ns: int,
    token_events: list[TokenEvent],
    num_prefill_tokens: int,
    num_decode_tokens: int,
    backend: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if arrival_time_ns < 0:
        raise ValueError(f"arrival_time_ns must be >= 0, got {arrival_time_ns}")

    token_rows: list[dict] = []
    prev_time_ns: int | None = None

    for idx, ev in enumerate(token_events):
        token_index = int(idx)
        token_time_ns = int(ev.token_time_ns)
        token_latency_ns = 0 if prev_time_ns is None else int(token_time_ns - prev_time_ns)
        prev_time_ns = token_time_ns

        row = {
            "request_id": int(request_id),
            "token_index": token_index,
            "token_time_ns": token_time_ns,
            "token_latency_ns": token_latency_ns,
        }
        if ev.token_id is not None:
            row["token_id"] = int(ev.token_id)
        token_rows.append(row)

    token_df = pd.DataFrame(token_rows)

    if token_events:
        first_token_time_ns = int(token_events[0].token_time_ns)
        completion_time_ns = int(token_events[-1].token_time_ns)
        ttft_ns = int(first_token_time_ns - arrival_time_ns)
        num_decode_tokens_actual = int(len(token_events))
        status = "ok"
    else:
        first_token_time_ns = int(arrival_time_ns)
        completion_time_ns = int(arrival_time_ns)
        ttft_ns = 0
        num_decode_tokens_actual = 0
        status = "error"

    request_row = {
        "request_id": int(request_id),
        "arrival_time_ns": int(arrival_time_ns),
        "first_token_time_ns": first_token_time_ns,
        "ttft_ns": ttft_ns,
        "completion_time_ns": completion_time_ns,
        "num_prefill_tokens": int(num_prefill_tokens),
        "num_decode_tokens": int(num_decode_tokens),
        "num_decode_tokens_actual": num_decode_tokens_actual,
        "status": status,
    }
    if backend is not None:
        request_row["backend"] = backend

    request_df = pd.DataFrame([request_row])
    return request_df, token_df


def write_run_outputs(
    out_dir: Path,
    *,
    request_df: pd.DataFrame,
    token_df: pd.DataFrame,
    run_meta: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        out_dir / "request_metrics.csv",
        request_df,
        required_columns=[
            "request_id",
            "arrival_time_ns",
            "first_token_time_ns",
            "ttft_ns",
            "num_prefill_tokens",
            "num_decode_tokens",
            "num_decode_tokens_actual",
            "status",
        ],
    )
    write_csv(
        out_dir / "token_metrics.csv",
        token_df,
        required_columns=[
            "request_id",
            "token_index",
            "token_time_ns",
            "token_latency_ns",
        ],
    )
    write_json(out_dir / "run_meta.json", run_meta)

