#!/usr/bin/env python3
"""
Sanitize sweep outputs for git tracking (remove machine-local absolute paths).

Inputs (in <sweep_dir>/):
  - comparison.md
  - comparison_scores.csv
  - comparison_runs.csv

Outputs (written to <expected_dir>/):
  - comparison.md
  - comparison_scores.csv
  - comparison_runs.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanitize a profile_method sweep snapshot for git tracking.")
    p.add_argument("--sweep-dir", required=True)
    p.add_argument("--expected-dir", required=True)
    return p.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _write_csv(path: Path, *, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    expected_dir = Path(args.expected_dir).expanduser().resolve()
    expected_dir.mkdir(parents=True, exist_ok=True)

    runs_csv = sweep_dir / "comparison_runs.csv"
    scores_csv = sweep_dir / "comparison_scores.csv"
    md_path = sweep_dir / "comparison.md"

    runs = _read_csv(runs_csv)
    replacements: dict[str, str] = {}
    for row in runs:
        method = row.get("method", "unknown")
        run_dir = row.get("run_dir", "")
        summary_md = row.get("summary_md", "")
        if run_dir:
            replacements[run_dir] = f"<RUN_DIR_{method}>"
        if summary_md:
            replacements[summary_md] = f"<SUMMARY_MD_{method}>"

    # comparison_runs.csv (replace run_dir and summary_md with placeholders)
    sanitized_runs: list[dict[str, Any]] = []
    for row in runs:
        out = dict(row)
        method = row.get("method", "unknown")
        out["run_dir"] = f"<RUN_DIR_{method}>"
        out["summary_md"] = f"<SUMMARY_MD_{method}>"
        sanitized_runs.append(out)
    _write_csv(
        expected_dir / "comparison_runs.csv",
        rows=sanitized_runs,
        fieldnames=list(runs[0].keys()) if runs else ["method", "run_dir", "summary_md"],
    )

    # comparison_scores.csv (copy verbatim; no machine-local paths)
    (expected_dir / "comparison_scores.csv").write_text(scores_csv.read_text(encoding="utf-8"), encoding="utf-8")

    # comparison.md (replace any known run_dir/summary paths; also normalize the "See ..." line)
    md = md_path.read_text(encoding="utf-8")
    for src, dst in replacements.items():
        md = md.replace(src, dst)
    md = md.replace(str(scores_csv), "comparison_scores.csv")
    (expected_dir / "comparison.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()

