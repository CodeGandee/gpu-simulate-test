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

