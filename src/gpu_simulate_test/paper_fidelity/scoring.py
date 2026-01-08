"""
Scoring utilities for paper-fidelity evaluation.

Given a simulation and real `request_metrics.csv`, compute percentile summaries and percent error:

    pct_error = abs(sim - real) / real

Thresholds default to Pass ≤ 5%, Warn 5–9%, Fail > 9%.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PAPER_FIDELITY_REQUEST_METRICS_REQUIRED_COLUMNS = [
    "request_id",
    "request_scheduling_delay",
    "request_execution_plus_preemption_time_normalized",
    "request_e2e_time_normalized",
    "request_num_decode_tokens",
]


@dataclass(frozen=True)
class ScoreThresholds:
    pass_pct: float = 0.05
    warn_pct: float = 0.09


@dataclass(frozen=True)
class ScoreResult:
    metric: str
    percentiles: list[float]
    sim: dict[float, float]
    real: dict[float, float]
    pct_error: dict[float, float]
    verdict: str  # "pass" | "warn" | "fail"


def load_metrics_csv(path: Path) -> pd.DataFrame:
    """Load request_metrics.csv and validate required columns."""
    df = pd.read_csv(path)
    missing = [c for c in PAPER_FIDELITY_REQUEST_METRICS_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")
    return df


def _parse_request_id(value: object) -> int:
    """Best-effort parsing for request ids across sim/real runners.

    Sarathi may emit IDs like "0_12" (replica prefix). Vidur emits ints.
    """
    s = str(value)
    if "_" in s:
        _, s = s.split("_", 1)
    try:
        return int(s)
    except ValueError as e:
        raise ValueError(f"Unexpected request_id value: {value!r}") from e


def validate_sim_vs_real_compatibility(*, sim_df: pd.DataFrame, real_df: pd.DataFrame) -> None:
    """Fail fast if sim and real metrics cannot be compared request-wise.

    Checks
    ------
    - `request_id` is unique in both datasets
    - sim and real contain the same request ids
    - `request_num_decode_tokens` match for every request id
    """
    sim_ids = sim_df["request_id"].map(_parse_request_id)
    real_ids = real_df["request_id"].map(_parse_request_id)

    if sim_ids.duplicated().any():
        dup = sim_ids.loc[sim_ids.duplicated()].unique()[:5].tolist()
        raise ValueError(f"sim metrics has duplicate request_id values (e.g. {dup}); request_id must be unique.")
    if real_ids.duplicated().any():
        dup = real_ids.loc[real_ids.duplicated()].unique()[:5].tolist()
        raise ValueError(f"real metrics has duplicate request_id values (e.g. {dup}); request_id must be unique.")

    sim_set = set(sim_ids.tolist())
    real_set = set(real_ids.tolist())
    missing = sorted(sim_set - real_set)[:5]
    extra = sorted(real_set - sim_set)[:5]
    if missing or extra:
        raise ValueError(
            "sim vs real request id sets differ (runs are not comparable): "
            f"missing_in_real={missing}, extra_in_real={extra}"
        )

    sim_tokens = pd.DataFrame(
        {
            "request_id": sim_ids.astype(int),
            "request_num_decode_tokens": pd.to_numeric(sim_df["request_num_decode_tokens"], errors="raise").astype(int),
        }
    )
    real_tokens = pd.DataFrame(
        {
            "request_id": real_ids.astype(int),
            "request_num_decode_tokens": pd.to_numeric(real_df["request_num_decode_tokens"], errors="raise").astype(int),
        }
    )

    merged = sim_tokens.merge(real_tokens, on="request_id", how="inner", suffixes=("_sim", "_real"))
    mismatches = merged.loc[merged["request_num_decode_tokens_sim"] != merged["request_num_decode_tokens_real"]]
    if len(mismatches) == 0:
        return

    sample = mismatches.head(10)
    details = "; ".join(
        f"id={int(rid)} sim={int(sim_n)} real={int(real_n)}"
        for rid, sim_n, real_n in zip(
            sample["request_id"].tolist(),
            sample["request_num_decode_tokens_sim"].tolist(),
            sample["request_num_decode_tokens_real"].tolist(),
        )
    )
    raise ValueError(
        "sim vs real request_num_decode_tokens mismatch (runs are not comparable). "
        f"First mismatches: {details}"
    )


def _percent_error(*, sim_value: float, real_value: float) -> float:
    if real_value == 0.0:
        if sim_value == 0.0:
            return 0.0
        return float("inf")
    return abs(sim_value - real_value) / abs(real_value)


def score_metric(
    *,
    sim_df: pd.DataFrame,
    real_df: pd.DataFrame,
    metric: str,
    percentiles: list[float],
    thresholds: ScoreThresholds,
) -> ScoreResult:
    """Compute percentile summaries and percent error for one metric."""
    if metric not in sim_df.columns:
        raise ValueError(f"sim metrics missing column: {metric}")
    if metric not in real_df.columns:
        raise ValueError(f"real metrics missing column: {metric}")

    sim_series = pd.to_numeric(sim_df[metric], errors="raise").astype(float).dropna()
    real_series = pd.to_numeric(real_df[metric], errors="raise").astype(float).dropna()
    if len(sim_series) == 0 or len(real_series) == 0:
        raise ValueError(f"metric {metric} has no finite values")

    qs = [float(q) for q in percentiles]
    for q in qs:
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"percentile must be in [0, 1] (got {q})")

    sim: dict[float, float] = {}
    real: dict[float, float] = {}
    err: dict[float, float] = {}

    for q in qs:
        sim_q = float(sim_series.quantile(q))
        real_q = float(real_series.quantile(q))
        sim[q] = sim_q
        real[q] = real_q
        err[q] = _percent_error(sim_value=sim_q, real_value=real_q)

    worst = float(np.nanmax(list(err.values())))
    if worst <= thresholds.pass_pct:
        verdict = "pass"
    elif worst <= thresholds.warn_pct:
        verdict = "warn"
    else:
        verdict = "fail"

    return ScoreResult(
        metric=metric,
        percentiles=qs,
        sim=sim,
        real=real,
        pct_error=err,
        verdict=verdict,
    )
