from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gpu_simulate_test.io import read_csv, read_json


@dataclass(frozen=True)
class RunMetrics:
    run_dir: Path
    request_metrics: pd.DataFrame
    token_metrics: pd.DataFrame
    run_meta: dict


def load_run_metrics(run_dir: Path) -> RunMetrics:
    run_dir = run_dir.expanduser().resolve()
    request_metrics = read_csv(
        run_dir / "request_metrics.csv",
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
        context="request_metrics",
    )
    token_metrics = read_csv(
        run_dir / "token_metrics.csv",
        required_columns=["request_id", "token_index", "token_time_ns", "token_latency_ns"],
        context="token_metrics",
    )
    run_meta = read_json(run_dir / "run_meta.json")
    return RunMetrics(
        run_dir=run_dir,
        request_metrics=request_metrics,
        token_metrics=token_metrics,
        run_meta=run_meta,
    )

