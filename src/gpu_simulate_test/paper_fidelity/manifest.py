"""
Manifest writer for paper-fidelity experiment batches.

This module records a small machine-readable index (`manifest.json`) for a batch run that
produces multiple paper-fidelity reports (e.g., static+dynamic across small/medium/full scales).

Expected input format
---------------------
The CLI reads a TSV file with the following header columns:

- workload: "static" | "dynamic"
- scale: "small" | "medium" | "full"
- scenario_name: output scenario name used by the run (folder name)
- report_dir: absolute or repo-relative path to the report directory
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_simulate_test.io import build_env_snapshot, get_git_info, utcnow_iso, write_json


@dataclass(frozen=True)
class ManifestRow:
    """One report entry in a paper-fidelity batch run."""

    workload: str
    scale: str
    scenario_name: str
    report_dir: Path

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        report_dir = self.report_dir.resolve()
        return {
            "workload": self.workload,
            "scale": self.scale,
            "scenario_name": self.scenario_name,
            "artifacts": {
                "report_dir": str(report_dir),
                "summary_md": str((report_dir / "summary.md").resolve()),
                "scores_json": str((report_dir / "scores.json").resolve()),
                "run_meta_json": str((report_dir / "run_meta.json").resolve()),
            },
        }


def _load_runs_tsv(path: Path) -> list[ManifestRow]:
    """Load a TSV file describing per-run report directories.

    Parameters
    ----------
    path
        TSV file containing run rows.

    Returns
    -------
    list[ManifestRow]
        Parsed run entries.
    """
    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"workload", "scale", "scenario_name", "report_dir"}
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing TSV header row")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: missing required TSV columns: {missing}")

        for i, row in enumerate(reader, start=2):
            workload = (row.get("workload") or "").strip()
            scale = (row.get("scale") or "").strip()
            scenario_name = (row.get("scenario_name") or "").strip()
            report_dir_raw = (row.get("report_dir") or "").strip()
            if not workload or not scale or not scenario_name or not report_dir_raw:
                raise ValueError(f"{path}:{i}: invalid row (empty required field): {row}")
            report_dir = Path(report_dir_raw).expanduser()
            rows.append(
                ManifestRow(
                    workload=workload,
                    scale=scale,
                    scenario_name=scenario_name,
                    report_dir=report_dir,
                )
            )
    return rows


def _validate_report_dirs(rows: list[ManifestRow], *, repo_root: Path) -> None:
    for row in rows:
        report_dir = row.report_dir.expanduser()
        if not report_dir.is_absolute():
            report_dir = (repo_root / report_dir).resolve()

        missing: list[str] = []
        for rel in ["summary.md", "scores.json", "run_meta.json"]:
            if not (report_dir / rel).exists():
                missing.append(rel)
        if missing:
            raise FileNotFoundError(f"{row.scenario_name}: missing report artifacts {missing} under {report_dir}")


def write_manifest(
    *,
    out_path: Path,
    repo_root: Path,
    base_scenario: str,
    run_id: str,
    rows: list[ManifestRow],
) -> Path:
    """Write a manifest JSON file and return its path.

    Parameters
    ----------
    out_path
        Where to write `manifest.json`.
    repo_root
        Repository root for provenance and resolving relative report paths.
    base_scenario
        Base scenario config name (e.g. `llama2_7b_arxiv`).
    run_id
        Caller-chosen identifier for the batch run.
    rows
        Per-run report entries.

    Returns
    -------
    pathlib.Path
        Path to the written manifest file.
    """
    git = get_git_info(repo_root=repo_root)
    payload: dict[str, Any] = {
        "schema_version": "v1",
        "generated_at": utcnow_iso(),
        "base_scenario": base_scenario,
        "run_id": run_id,
        "git_commit": git.commit or "unknown",
        "git_dirty": git.dirty,
        "env": build_env_snapshot(),
        "runs": [row.to_dict() for row in rows],
    }
    write_json(out_path, payload)
    return out_path


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for writing a paper-fidelity batch manifest."""
    parser = argparse.ArgumentParser(prog="paper-fidelity-manifest")
    parser.add_argument("--runs-tsv", required=True, help="TSV file produced by a batch runner script")
    parser.add_argument("--out", required=True, help="Output manifest.json path")
    parser.add_argument("--base-scenario", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", default=".", help="Repo root for resolving relative paths (default: CWD)")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    rows = _load_runs_tsv(Path(args.runs_tsv).expanduser())
    _validate_report_dirs(rows, repo_root=repo_root)
    write_manifest(
        out_path=Path(args.out).expanduser(),
        repo_root=repo_root,
        base_scenario=str(args.base_scenario),
        run_id=str(args.run_id),
        rows=rows,
    )


if __name__ == "__main__":
    main()

