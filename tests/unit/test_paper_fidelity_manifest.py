from __future__ import annotations

from pathlib import Path

from gpu_simulate_test.paper_fidelity.manifest import main


def test_manifest_writer_creates_json(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.md").write_text("# summary\n", encoding="utf-8")
    (report_dir / "scores.json").write_text("{}", encoding="utf-8")
    (report_dir / "run_meta.json").write_text("{}", encoding="utf-8")

    runs_tsv = tmp_path / "runs.tsv"
    runs_tsv.write_text(
        "workload\tscale\tscenario_name\treport_dir\n"
        f"static\tsmall\tunit_test\t{report_dir}\n",
        encoding="utf-8",
    )

    out = tmp_path / "manifest.json"
    main(
        [
            "--runs-tsv",
            str(runs_tsv),
            "--out",
            str(out),
            "--base-scenario",
            "llama2_7b_arxiv",
            "--run-id",
            "unit_test_run",
            "--repo-root",
            str(tmp_path),
        ]
    )

    assert out.exists()

