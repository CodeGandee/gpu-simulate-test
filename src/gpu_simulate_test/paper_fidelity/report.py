"""
Report writer for paper-fidelity workflows.

Writes a human-readable `summary.md` under:

    results/reports/<date>/paper_fidelity/<scenario_name>/
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gpu_simulate_test.io import write_json
from gpu_simulate_test.paper_fidelity.scoring import ScoreResult, load_metrics_csv


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


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_ecdf_svg(*, sim_values: np.ndarray, real_values: np.ndarray, metric: str, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to write SVG plots for paper-fidelity reports; run inside the Pixi env."
        ) from e

    def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.sort(x)
        if len(x) == 0:
            return x, x
        y = np.linspace(1.0 / len(x), 1.0, num=len(x), endpoint=True)
        return x, y

    sim_x, sim_y = _ecdf(sim_values)
    real_x, real_y = _ecdf(real_values)

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=150)
    ax.plot(real_x, real_y, label="real", linewidth=2)
    ax.plot(sim_x, sim_y, label="sim", linewidth=2)
    ax.set_title(f"ECDF: {metric}")
    ax.set_xlabel("normalized latency")
    ax.set_ylabel("fraction of requests")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _write_percentiles_svg(
    *,
    metric: str,
    percentiles: list[float],
    sim: dict[float, float],
    real: dict[float, float],
    paper_by_label: dict[str, float] | None,
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to write SVG plots for paper-fidelity reports; run inside the Pixi env."
        ) from e

    labels = [f"p{int(q * 100)}" for q in percentiles]
    sim_vals = [float(sim[q]) for q in percentiles]
    real_vals = [float(real[q]) for q in percentiles]
    has_paper = bool(paper_by_label)
    paper_vals = [float(paper_by_label.get(lbl)) for lbl in labels] if paper_by_label else []

    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=150)
    if has_paper:
        width = 0.25
        ax.bar(x - width, paper_vals, width=width, label="paper")
        ax.bar(x, real_vals, width=width, label="real")
        ax.bar(x + width, sim_vals, width=width, label="sim")
    else:
        width = 0.35
        ax.bar(x - width / 2, real_vals, width=width, label="real")
        ax.bar(x + width / 2, sim_vals, width=width, label="sim")

    ax.set_title(f"Percentiles: {metric}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("normalized latency")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def write_summary_md(*, inputs: ReportInputs, results: list[ScoreResult], meta: dict) -> Path:
    """Write summary.md and return its path."""
    inputs.out_dir.mkdir(parents=True, exist_ok=True)

    summary_md = inputs.out_dir / "summary.md"
    tables_dir = inputs.out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = inputs.out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

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

    # Machine-readable scores for programmatic analysis.
    scores: dict[str, object] = {
        "schema_version": "v1",
        "scenario_name": inputs.scenario_name,
        "generated_at": _utcnow_iso(),
        "inputs": {"sim_csv": str(inputs.sim_csv), "real_csv": str(inputs.real_csv)},
        "metrics": [],
    }
    if paper_values:
        scores["paper_reference"] = meta.get("paper_reference")
    for r in results:
        per: dict[str, object] = {}
        for q in r.percentiles:
            percentile = f"p{int(q * 100)}"
            entry: dict[str, object] = {
                "sim": float(r.sim[q]),
                "real": float(r.real[q]),
                "pct_error": float(r.pct_error[q]),
            }
            if paper_values:
                paper_value = paper_values.get((r.metric, percentile))
                if paper_value is not None:
                    entry["paper"] = float(paper_value)
                    entry["sim_vs_paper_pct_error"] = float(_percent_error(sim_value=float(r.sim[q]), ref_value=float(paper_value)))
            per[percentile] = entry
        scores["metrics"].append(
            {
                "metric": r.metric,
                "verdict": r.verdict,
                "percentiles": per,
            }
        )
    write_json(inputs.out_dir / "scores.json", scores)

    # Figures: sim vs real ECDFs for the two paper-facing normalized metrics.
    lines.append("## Figures")
    try:
        sim_df = load_metrics_csv(inputs.sim_csv)
        real_df = load_metrics_csv(inputs.real_csv)

        figure_specs = [
            ("Static normalized latency", "request_execution_plus_preemption_time_normalized"),
            ("Dynamic normalized latency", "request_e2e_time_normalized"),
        ]
        for title, metric in figure_specs:
            sim_values = np.asarray(sim_df[metric], dtype=float)
            real_values = np.asarray(real_df[metric], dtype=float)
            sim_values = sim_values[np.isfinite(sim_values)]
            real_values = real_values[np.isfinite(real_values)]
            if len(sim_values) == 0 or len(real_values) == 0:
                continue

            out_svg = figs_dir / f"{metric}_ecdf.svg"
            _write_ecdf_svg(sim_values=sim_values, real_values=real_values, metric=metric, out_path=out_svg)

            score = next((r for r in results if r.metric == metric), None)
            pct_svg = None
            if score is not None:
                paper_by_label = None
                if paper_values:
                    paper_by_label = {
                        f"p{int(q * 100)}": float(paper_values[(metric, f"p{int(q * 100)}")])
                        for q in score.percentiles
                        if (metric, f"p{int(q * 100)}") in paper_values
                    }
                pct_svg = figs_dir / f"{metric}_percentiles.svg"
                _write_percentiles_svg(
                    metric=metric,
                    percentiles=score.percentiles,
                    sim=score.sim,
                    real=score.real,
                    paper_by_label=paper_by_label,
                    out_path=pct_svg,
                )

            lines.append(f"### {title}")
            lines.append(f"![ECDF: {metric}](figs/{out_svg.name})")
            if pct_svg is not None:
                lines.append(f"![Percentiles: {metric}](figs/{pct_svg.name})")
            lines.append("")
    except Exception as e:
        lines.append(f"- Failed to generate figures: {type(e).__name__}: {e}")
        lines.append("")

    if any(r.verdict in {"warn", "fail"} for r in results):
        lines.append("## Gap Diagnosis")
        for item in diagnose_gap(sim_csv=inputs.sim_csv, real_csv=inputs.real_csv, sim_meta=meta):
            lines.append(f"- {item}")
        lines.append("")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_md
