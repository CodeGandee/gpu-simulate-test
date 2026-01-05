from __future__ import annotations

from pathlib import Path

import pytest

from gpu_simulate_test.paper_fidelity.scoring import (
    ScoreThresholds,
    load_metrics_csv,
    score_metric,
)


def _fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "paper_fidelity"


def test_scorer_percent_error_matches_hand_calc() -> None:
    sim_df = load_metrics_csv(_fixtures_dir() / "sim_request_metrics.csv")
    real_df = load_metrics_csv(_fixtures_dir() / "real_request_metrics.csv")

    result = score_metric(
        sim_df=sim_df,
        real_df=real_df,
        metric="request_e2e_time_normalized",
        percentiles=[0.5, 0.95],
        thresholds=ScoreThresholds(pass_pct=0.05, warn_pct=0.09),
    )

    assert result.sim[0.5] == 105.0
    assert result.real[0.5] == 100.0
    assert result.pct_error[0.5] == pytest.approx(0.05)
    assert result.pct_error[0.95] == pytest.approx(0.05)
    assert result.verdict == "pass"


def test_threshold_verdicts_pass_warn_fail() -> None:
    sim_df = load_metrics_csv(_fixtures_dir() / "sim_request_metrics.csv")
    real_df = load_metrics_csv(_fixtures_dir() / "real_request_metrics.csv")

    thresholds = ScoreThresholds(pass_pct=0.05, warn_pct=0.09)

    pass_result = score_metric(
        sim_df=sim_df,
        real_df=real_df,
        metric="request_e2e_time_normalized",
        percentiles=[0.5],
        thresholds=thresholds,
    )
    assert pass_result.verdict == "pass"

    fail_result = score_metric(
        sim_df=sim_df,
        real_df=real_df,
        metric="request_execution_plus_preemption_time_normalized",
        percentiles=[0.5],
        thresholds=thresholds,
    )
    assert fail_result.verdict == "fail"

