from __future__ import annotations

from pathlib import Path

from gpu_simulate_test.real_bench.backends.sarathi_paper_fidelity_backend import (
    convert_sequence_metrics_to_request_metrics,
)


def test_sequence_metrics_conversion_emits_required_columns() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixture = repo_root / "tests" / "fixtures" / "paper_fidelity" / "sarathi_sequence_metrics.csv"

    df = convert_sequence_metrics_to_request_metrics(fixture)
    assert "request_id" in df.columns

    for c in [
        "request_scheduling_delay",
        "request_execution_plus_preemption_time_normalized",
        "request_e2e_time_normalized",
        "request_num_decode_tokens",
    ]:
        assert c in df.columns

