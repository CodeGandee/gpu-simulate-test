from __future__ import annotations

from pathlib import Path

import pandas as pd

from gpu_simulate_test.paper_fidelity.report import ReportInputs, write_summary_md
from gpu_simulate_test.paper_fidelity.scoring import ScoreThresholds, load_metrics_csv, score_metric


def _write_metrics_csv(path: Path, *, values: list[float]) -> None:
    df = pd.DataFrame(
        {
            "request_id": list(range(len(values))),
            "request_scheduling_delay": [0.0] * len(values),
            "request_execution_plus_preemption_time_normalized": values,
            "request_e2e_time_normalized": values,
            "request_num_decode_tokens": [1] * len(values),
        }
    )
    df.to_csv(path, index=False)


def test_paper_fidelity_report_writes_json_and_svgs(tmp_path: Path) -> None:
    sim_csv = tmp_path / "sim.csv"
    real_csv = tmp_path / "real.csv"
    _write_metrics_csv(sim_csv, values=[1.0, 1.1, 1.2, 1.3])
    _write_metrics_csv(real_csv, values=[1.05, 1.05, 1.25, 1.35])

    sim_df = load_metrics_csv(sim_csv)
    real_df = load_metrics_csv(real_csv)
    results = [
        score_metric(
            sim_df=sim_df,
            real_df=real_df,
            metric="request_execution_plus_preemption_time_normalized",
            percentiles=[0.5, 0.95],
            thresholds=ScoreThresholds(pass_pct=0.5, warn_pct=0.9),
        ),
        score_metric(
            sim_df=sim_df,
            real_df=real_df,
            metric="request_e2e_time_normalized",
            percentiles=[0.5, 0.95],
            thresholds=ScoreThresholds(pass_pct=0.5, warn_pct=0.9),
        ),
    ]

    out_dir = tmp_path / "report"
    write_summary_md(
        inputs=ReportInputs(scenario_name="unit_test", sim_csv=sim_csv, real_csv=real_csv, out_dir=out_dir),
        results=results,
        meta={"schema_version": "v1"},
    )

    assert (out_dir / "summary.md").exists()
    assert (out_dir / "run_meta.json").exists()
    assert (out_dir / "scores.json").exists()
    assert (out_dir / "figs" / "request_execution_plus_preemption_time_normalized_ecdf.svg").exists()
    assert (out_dir / "figs" / "request_e2e_time_normalized_ecdf.svg").exists()
    assert (out_dir / "figs" / "request_execution_plus_preemption_time_normalized_percentiles.svg").exists()
    assert (out_dir / "figs" / "request_e2e_time_normalized_percentiles.svg").exists()
