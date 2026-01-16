#!/usr/bin/env python3
"""
Summarize a `profiling.mlp.profile_method` sweep into a small comparison report.

Expected input layout (produced by run_sweep_static_profile_methods.sh):

    <sweep_dir>/
      cuda_event_summary.md
      record_function_summary.md
      kineto_summary.md
      perf_counter_summary.md
      cuda_event_run_dir.txt
      ...

Outputs (written to <sweep_dir>/):

  - comparison.md
  - comparison_scores.csv
  - comparison_runs.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable


_SCORE_LINE_RE = re.compile(
    r"^\|\s*(?P<metric>[^|]+?)\s*\|\s*(?P<pct>p\d+)\s*\|\s*(?P<sim>[^|]+?)\s*\|\s*(?P<real>[^|]+?)\s*\|\s*(?P<err>[^|]+?)\s*\|\s*$"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize a vidur-cli profile_method sweep.")
    p.add_argument("--sweep-dir", required=True, help="Path written by the sweep runner (under tmp/).")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["cuda_event", "record_function", "kineto", "perf_counter"],
        help="Methods to summarize (defaults to all 4 Vidur choices).",
    )
    return p.parse_args()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_one_line(path: Path) -> str:
    return _read_text(path).strip()


def _parse_mlp_block(summary_text: str) -> tuple[str, str, str]:
    """Return (profile_method, validation, fallback) from the `- mlp:` section (best-effort)."""
    in_mlp = False
    mlp_profile_method = ""
    mlp_validation = ""
    mlp_fallback = ""
    for line in summary_text.splitlines():
        if line.strip() == "- mlp:":
            in_mlp = True
            continue
        if in_mlp and (line.startswith("## ") or (line.startswith("-") and not line.startswith("  -"))):
            in_mlp = False
        if not in_mlp:
            continue
        if "profile_method:" in line:
            mlp_profile_method = line.split("`", 2)[1] if "`" in line else line.strip()
        if "validation:" in line:
            mlp_validation = line.split("`", 2)[1] if "`" in line else line.strip()
        if "fallback:" in line:
            mlp_fallback = line.split("`", 2)[1] if "`" in line else line.strip()
    return mlp_profile_method, mlp_validation, mlp_fallback


def _parse_scores(summary_text: str, *, method: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    in_scores = False
    for line in summary_text.splitlines():
        if line.strip() == "## Scores":
            in_scores = True
            continue
        if in_scores and line.startswith("## "):
            break
        if not in_scores or not line.startswith("|"):
            continue
        if line.startswith("| Metric") or line.startswith("|--------"):
            continue
        m = _SCORE_LINE_RE.match(line)
        if not m:
            continue
        out.append(
            {
                "method": method,
                "metric": m.group("metric").strip(),
                "percentile": m.group("pct").strip(),
                "sim": m.group("sim").strip(),
                "real": m.group("real").strip(),
                "percent_error": m.group("err").strip(),
            }
        )
    return out


def _write_csv(path: Path, *, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(list(rows))


def main() -> None:
    args = _parse_args()
    sweep_dir = Path(args.sweep_dir).expanduser().resolve()
    methods = [str(m) for m in args.methods]

    score_rows: list[dict[str, str]] = []
    run_rows: list[dict[str, str]] = []

    for method in methods:
        summary_path = sweep_dir / f"{method}_summary.md"
        run_dir_txt = sweep_dir / f"{method}_run_dir.txt"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary file: {summary_path}")
        if not run_dir_txt.exists():
            raise FileNotFoundError(f"Missing run_dir file: {run_dir_txt}")

        summary_text = _read_text(summary_path)
        run_dir = _read_one_line(run_dir_txt)

        mlp_profile_method, mlp_validation, mlp_fallback = _parse_mlp_block(summary_text)

        score_rows.extend(_parse_scores(summary_text, method=method))
        run_rows.append(
            {
                "method": method,
                "run_dir": run_dir,
                "summary_md": str(summary_path),
                "mlp_profile_method": mlp_profile_method,
                "mlp_validation": mlp_validation,
                "mlp_fallback": mlp_fallback,
            }
        )

    scores_csv = sweep_dir / "comparison_scores.csv"
    runs_csv = sweep_dir / "comparison_runs.csv"
    _write_csv(
        scores_csv,
        rows=score_rows,
        fieldnames=["method", "metric", "percentile", "sim", "real", "percent_error"],
    )
    _write_csv(
        runs_csv,
        rows=run_rows,
        fieldnames=["method", "run_dir", "summary_md", "mlp_profile_method", "mlp_validation", "mlp_fallback"],
    )

    # Small human-readable table: selected metrics only.
    by_method: dict[str, dict[tuple[str, str], str]] = {}
    for row in score_rows:
        by_method.setdefault(row["method"], {})[(row["metric"], row["percentile"])] = row["percent_error"]

    focus = [
        ("request_execution_plus_preemption_time_normalized", "p50"),
        ("request_execution_plus_preemption_time_normalized", "p95"),
        ("request_e2e_time_normalized", "p50"),
        ("request_e2e_time_normalized", "p95"),
    ]

    md_lines: list[str] = []
    md_lines.append(f"# Profile method sweep ({sweep_dir.name})")
    md_lines.append("")
    md_lines.append("## Runs")
    md_lines.append("| method | run_dir |")
    md_lines.append("|--------|---------|")
    for row in run_rows:
        md_lines.append(f"| {row['method']} | `{row['run_dir']}` |")
    md_lines.append("")
    md_lines.append("## Percent error (selected metrics)")
    md_lines.append("| method | exec+pree p50 | exec+pree p95 | e2e p50 | e2e p95 |")
    md_lines.append("|--------|--------------:|--------------:|--------:|--------:|")
    for method in methods:
        def get(metric: str, pct: str) -> str:
            return by_method.get(method, {}).get((metric, pct), "")

        md_lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    get(focus[0][0], focus[0][1]),
                    get(focus[1][0], focus[1][1]),
                    get(focus[2][0], focus[2][1]),
                    get(focus[3][0], focus[3][1]),
                ]
            )
            + " |"
        )
    md_lines.append("")
    md_lines.append("## Full score table")
    md_lines.append(f"See `{scores_csv}`")
    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append("- Each run is a full `init-run → trace → profile → sim → real → report` pipeline.")
    md_lines.append("- `kineto` is expected to be slow due to profiler overhead.")

    (sweep_dir / "comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

