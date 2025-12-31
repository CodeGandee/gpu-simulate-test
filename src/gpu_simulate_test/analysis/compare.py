from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gpu_simulate_test.analysis.load_metrics import RunMetrics


@dataclass(frozen=True)
class CompareResult:
    ttft_table: pd.DataFrame
    token_latency_table: pd.DataFrame


def align_tokens_by_actual_decode(*, real: RunMetrics, sim: RunMetrics) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_actual = real.request_metrics[["request_id", "num_decode_tokens_actual"]].copy()
    real_actual["num_decode_tokens_actual"] = real_actual["num_decode_tokens_actual"].astype(int)

    def _truncate(tokens: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(tokens, real_actual, on=["request_id"], how="inner")
        merged = merged[merged["token_index"].astype(int) < merged["num_decode_tokens_actual"]]
        merged = merged.drop(columns=["num_decode_tokens_actual"])
        return merged.sort_values(["request_id", "token_index"]).reset_index(drop=True)

    return _truncate(real.token_metrics), _truncate(sim.token_metrics)


def compute_percentiles(series: pd.Series, percentiles: list[float]) -> pd.Series:
    q = series.quantile(percentiles, interpolation="linear")
    q.index = [f"p{int(p * 100)}" for p in percentiles]
    return q

