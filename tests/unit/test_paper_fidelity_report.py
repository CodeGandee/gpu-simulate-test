from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gpu_simulate_test.paper_fidelity.report import ReportInputs, regenerate_summary_md_from_report_dir, write_summary_md
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
    assert (out_dir / "inputs" / "sim_request_metrics.csv").exists()
    assert (out_dir / "inputs" / "real_request_metrics.csv").exists()
    assert (out_dir / "figs" / "request_execution_plus_preemption_time_normalized_ecdf.svg").exists()
    assert (out_dir / "figs" / "request_e2e_time_normalized_ecdf.svg").exists()
    assert (out_dir / "figs" / "request_execution_plus_preemption_time_normalized_percentiles.svg").exists()
    assert (out_dir / "figs" / "request_e2e_time_normalized_percentiles.svg").exists()


def test_paper_fidelity_report_regeneration_only_rewrites_md(tmp_path: Path) -> None:
    report_dir = tmp_path / "report_dir"
    report_dir.mkdir(parents=True, exist_ok=True)

    meta_path = report_dir / "run_meta.json"
    scores_path = report_dir / "scores.json"
    summary_path = report_dir / "summary.md"

    meta = {"schema_version": "v1", "scenario_name": "unit_test"}
    scores = {
        "schema_version": "v1",
        "scenario_name": "unit_test",
        "generated_at": "2026-01-01T00:00:00Z",
        "inputs": {"sim_csv": "/abs/sim.csv", "real_csv": "/abs/real.csv"},
        "metrics": [
            {
                "metric": "request_execution_plus_preemption_time_normalized",
                "verdict": "pass",
                "percentiles": {
                    "p50": {"sim": 1.0, "real": 1.1, "pct_error": 0.090909},
                },
            }
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scores_path.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text("old\n", encoding="utf-8")

    meta_before = meta_path.read_text(encoding="utf-8")
    scores_before = scores_path.read_text(encoding="utf-8")

    regenerate_summary_md_from_report_dir(report_dir)

    assert meta_path.read_text(encoding="utf-8") == meta_before
    assert scores_path.read_text(encoding="utf-8") == scores_before

    summary = summary_path.read_text(encoding="utf-8")
    assert "# Paper Fidelity Report: unit_test" in summary
    assert "## Scores" in summary
    assert "request_execution_plus_preemption_time_normalized" in summary


def test_paper_fidelity_report_regeneration_can_omit_paper_reference(tmp_path: Path) -> None:
    report_dir = tmp_path / "report_dir"
    report_dir.mkdir(parents=True, exist_ok=True)

    meta_path = report_dir / "run_meta.json"
    scores_path = report_dir / "scores.json"

    paper_reference = {
        "schema_version": "v1",
        "requested": True,
        "matched": True,
        "workload_mode": "static",
        "load_frac_of_capacity": None,
        "criteria": {
            "metric": "request_execution_plus_preemption_time_normalized",
            "model": "llama2-7b",
            "trace": "arxiv",
            "series": "predicted",
            "p50_json": "paper_p50.json",
            "p95_json": "paper_p95.json",
        },
        "rows": [
            {
                "metric": "request_execution_plus_preemption_time_normalized",
                "percentile": "p50",
                "value": 0.9,
                "source_json": "paper_p50.json",
            }
        ],
    }

    meta = {"schema_version": "v1", "scenario_name": "unit_test", "paper_reference": paper_reference}
    scores = {
        "schema_version": "v1",
        "scenario_name": "unit_test",
        "generated_at": "2026-01-01T00:00:00Z",
        "inputs": {"sim_csv": "/abs/sim.csv", "real_csv": "/abs/real.csv"},
        "paper_reference": paper_reference,
        "metrics": [
            {
                "metric": "request_execution_plus_preemption_time_normalized",
                "verdict": "pass",
                "percentiles": {
                    "p50": {"sim": 1.0, "real": 1.1, "pct_error": 0.090909, "paper": 0.9, "sim_vs_paper_pct_error": 0.111111},
                },
            }
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scores_path.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    regenerate_summary_md_from_report_dir(report_dir, include_paper_reference=True)
    summary_with_paper = (report_dir / "summary.md").read_text(encoding="utf-8")
    assert "## Paper Reference" in summary_with_paper
    assert "| Metric | Percentile | Paper | Sim | Real |" in summary_with_paper

    regenerate_summary_md_from_report_dir(report_dir, include_paper_reference=False)
    summary_without_paper = (report_dir / "summary.md").read_text(encoding="utf-8")
    assert "## Paper Reference" not in summary_without_paper
    assert "| Metric | Percentile | Paper | Sim | Real |" not in summary_without_paper
    assert "| Metric | Percentile | Sim | Real |" in summary_without_paper
