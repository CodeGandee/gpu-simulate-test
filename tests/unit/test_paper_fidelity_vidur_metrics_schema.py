from __future__ import annotations

from pathlib import Path

import pandas as pd

from gpu_simulate_test.vidur_ext.sim_runner import convert_vidur_request_metrics_to_paper_fidelity


def test_vidur_request_metrics_preserves_normalized_columns() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixture = repo_root / "tests" / "fixtures" / "paper_fidelity" / "vidur_request_metrics_raw.csv"

    raw = pd.read_csv(fixture)
    out = convert_vidur_request_metrics_to_paper_fidelity(fixture)

    assert "request_id" in out.columns
    assert "Request Id" not in out.columns

    for c in [
        "request_scheduling_delay",
        "request_execution_plus_preemption_time_normalized",
        "request_e2e_time_normalized",
        "request_num_decode_tokens",
    ]:
        assert c in out.columns

    assert out["request_id"].tolist() == raw["Request Id"].tolist()
    assert out["request_e2e_time_normalized"].tolist() == raw["request_e2e_time_normalized"].tolist()
    assert (
        out["request_execution_plus_preemption_time_normalized"].tolist()
        == raw["request_execution_plus_preemption_time_normalized"].tolist()
    )

