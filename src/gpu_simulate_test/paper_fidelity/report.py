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
        "Vidur sim may underpredict wall-clock latency when CPU/runtime overhead is excluded and/or "
        "the profiling bundle is not host-matched; see `context/issues/known/issue-vidur-sim-underpredicts-sarathi-real.md`."
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


def _percent_error(*, sim_value: float, ref_value: float) -> float:
    if ref_value == 0.0:
        if sim_value == 0.0:
            return 0.0
        return float("inf")
    return abs(sim_value - ref_value) / abs(ref_value)


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

    paper_reference = meta.get("paper_reference")
    paper_values: dict[tuple[str, str], float] = {}
    paper_rows: list[dict] = []
    if isinstance(paper_reference, dict):
        rows = paper_reference.get("rows")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                metric = row.get("metric")
                percentile = row.get("percentile")
                value = row.get("value")
                if not isinstance(metric, str) or not isinstance(percentile, str):
                    continue
                try:
                    paper_values[(metric, percentile)] = float(value)
                except (TypeError, ValueError):
                    continue
                paper_rows.append(row)

    if paper_values:
        first = paper_rows[0] if paper_rows else {}
        model = first.get("model")
        trace = first.get("trace")
        series = first.get("series")
        load_frac = paper_reference.get("load_frac_of_capacity") if isinstance(paper_reference, dict) else None

        sources: list[str] = []
        for row in paper_rows:
            source_json = row.get("source_json")
            if isinstance(source_json, str) and source_json not in sources:
                sources.append(source_json)

        lines.append("## Paper Reference")
        if isinstance(model, str):
            lines.append(f"- model: `{model}`")
        if isinstance(trace, str):
            lines.append(f"- trace: `{trace}`")
        if isinstance(series, str):
            lines.append(f"- series: `{series}`")
        if load_frac is not None:
            lines.append(f"- load_frac_of_capacity: `{load_frac}`")
        if sources:
            lines.append(f"- sources: {', '.join([f'`{s}`' for s in sources])}")
        lines.append("")

    lines.append("## Scores")
    if paper_values:
        lines.append("| Metric | Percentile | Paper | Sim | Real | Sim vs Paper | Sim vs Real | Verdict |")
        lines.append("|--------|------------|-------|-----|------|--------------|-------------|---------|")
    else:
        lines.append("| Metric | Percentile | Sim | Real | Percent error | Verdict |")
        lines.append("|--------|------------|-----|------|---------------|---------|")
    for r in results:
        for q in r.percentiles:
            percentile = f"p{int(q * 100)}"
            if paper_values:
                paper_value = paper_values.get((r.metric, percentile))
                if paper_value is None:
                    lines.append(
                        f"| {r.metric} | {percentile} | N/A | {r.sim[q]:.6g} | {r.real[q]:.6g} | N/A | {_fmt_pct(r.pct_error[q])} | {r.verdict} |"
                    )
                    continue
                sim_vs_paper = _percent_error(sim_value=float(r.sim[q]), ref_value=float(paper_value))
                lines.append(
                    f"| {r.metric} | {percentile} | {paper_value:.6g} | {r.sim[q]:.6g} | {r.real[q]:.6g} | {_fmt_pct(sim_vs_paper)} | {_fmt_pct(r.pct_error[q])} | {r.verdict} |"
                )
            else:
                lines.append(
                    f"| {r.metric} | {percentile} | {r.sim[q]:.6g} | {r.real[q]:.6g} | {_fmt_pct(r.pct_error[q])} | {r.verdict} |"
                )

    lines.append("")

    if any(r.verdict in {"warn", "fail"} for r in results):
        lines.append("## Gap Diagnosis")
        for item in diagnose_gap(sim_csv=inputs.sim_csv, real_csv=inputs.real_csv, sim_meta=meta):
            lines.append(f"- {item}")
        lines.append("")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_md
