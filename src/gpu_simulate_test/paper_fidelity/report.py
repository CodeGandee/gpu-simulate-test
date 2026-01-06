"""
Report writer for paper-fidelity workflows.

Writes a human-readable `summary.md` under:

    results/reports/<date>/paper_fidelity/<scenario_name>/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gpu_simulate_test.io import write_json
from gpu_simulate_test.paper_fidelity.scoring import ScoreResult


@dataclass(frozen=True)
class ReportInputs:
    scenario_name: str
    sim_csv: Path
    real_csv: Path
    out_dir: Path


def diagnose_gap(*, sim_csv: Path, real_csv: Path, sim_meta: dict | None) -> list[str]:
    """Return a short list of hypotheses with evidence pointers."""
    hypotheses: list[str] = []

    hypotheses.append(
        "Vidur sim may underpredict wall-clock latency when CPU/runtime overhead is excluded; "
        "see `context/issues/known/issue-vidur-sim-underpredicts-sarathi-real-qwen3-0.6b.md`."
    )

    if sim_meta and sim_meta.get("vidur_raw_dir"):
        hypotheses.append(
            "Inspect raw simulator outputs under the recorded `vidur_raw_dir` to verify the presence and "
            "distribution of normalized metric columns."
        )

    hypotheses.append(f"Sim metrics: `{sim_csv}`; Real metrics: `{real_csv}`.")
    return hypotheses


def _fmt_pct(x: float) -> str:
    if x == float("inf"):
        return "inf"
    return f"{x * 100:.2f}%"


def write_summary_md(*, inputs: ReportInputs, results: list[ScoreResult], meta: dict) -> Path:
    """Write summary.md and return its path."""
    inputs.out_dir.mkdir(parents=True, exist_ok=True)

    summary_md = inputs.out_dir / "summary.md"
    tables_dir = inputs.out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    write_json(inputs.out_dir / "run_meta.json", meta)

    lines: list[str] = []
    lines.append(f"# Paper Fidelity Report: {inputs.scenario_name}")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- sim: `{inputs.sim_csv}`")
    lines.append(f"- real: `{inputs.real_csv}`")
    lines.append("")

    profiling = meta.get("profiling")
    if isinstance(profiling, dict):
        root = profiling.get("root")
        mode = profiling.get("mode")
        interpretation = profiling.get("interpretation")

        lines.append("## Profiling")
        if root:
            lines.append(f"- root: `{root}`")
        if mode:
            lines.append(f"- mode: `{mode}`")
        if interpretation:
            lines.append(f"- interpretation: {interpretation}")
        lines.append("")

    lines.append("## Scores")
    lines.append("| Metric | Percentile | Sim | Real | Percent error | Verdict |")
    lines.append("|--------|------------|-----|------|---------------|---------|")
    for r in results:
        for q in r.percentiles:
            lines.append(
                f"| {r.metric} | p{int(q * 100)} | {r.sim[q]:.6g} | {r.real[q]:.6g} | {_fmt_pct(r.pct_error[q])} | {r.verdict} |"
            )

    lines.append("")

    if any(r.verdict in {"warn", "fail"} for r in results):
        lines.append("## Gap Diagnosis")
        for item in diagnose_gap(sim_csv=inputs.sim_csv, real_csv=inputs.real_csv, sim_meta=meta):
            lines.append(f"- {item}")
        lines.append("")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_md
