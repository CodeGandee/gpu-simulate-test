"""
Report writer for paper-fidelity workflows.

Writes a human-readable `summary.md` under:

    results/reports/<date>/paper_fidelity/<scenario_name>/
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gpu_simulate_test.io import read_json, write_json
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


def _parse_percentile_label(label: str) -> float:
    if not label.startswith("p"):
        raise ValueError(f"Invalid percentile label: {label!r} (expected 'p50', 'p99', etc.)")
    suffix = label[1:]
    if not suffix.isdigit():
        raise ValueError(f"Invalid percentile label: {label!r} (expected digits after 'p')")
    value = int(suffix)
    if not (0 <= value <= 100):
        raise ValueError(f"Invalid percentile label: {label!r} (expected p0..p100)")
    return value / 100.0


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


def load_score_results_from_scores_json(scores: dict[str, Any]) -> list[ScoreResult]:
    """Parse `scores.json` into `ScoreResult` records.

    Parameters
    ----------
    scores
        Parsed JSON dictionary produced by `write_summary_md`.

    Returns
    -------
    list[ScoreResult]
        A list of metric results suitable for report regeneration.
    """
    metrics_val = scores.get("metrics")
    if not isinstance(metrics_val, list):
        raise ValueError("scores.json: missing or invalid `metrics` list")

    results: list[ScoreResult] = []
    for idx, metric_entry in enumerate(metrics_val):
        if not isinstance(metric_entry, dict):
            raise ValueError(f"scores.json: metrics[{idx}] must be an object")

        metric = metric_entry.get("metric")
        verdict = metric_entry.get("verdict")
        percentiles_val = metric_entry.get("percentiles")
        if not isinstance(metric, str):
            raise ValueError(f"scores.json: metrics[{idx}].metric must be a string")
        if not isinstance(verdict, str):
            raise ValueError(f"scores.json: metrics[{idx}].verdict must be a string")
        if not isinstance(percentiles_val, dict):
            raise ValueError(f"scores.json: metrics[{idx}].percentiles must be an object")

        percentiles: list[float] = []
        sim: dict[float, float] = {}
        real: dict[float, float] = {}
        pct_error: dict[float, float] = {}

        for label, values in percentiles_val.items():
            if not isinstance(label, str):
                raise ValueError(f"scores.json: metrics[{idx}].percentiles keys must be strings")
            if not isinstance(values, dict):
                raise ValueError(f"scores.json: metrics[{idx}].percentiles[{label}] must be an object")

            q = _parse_percentile_label(label)
            try:
                sim[q] = float(values["sim"])
                real[q] = float(values["real"])
                pct_error[q] = float(values["pct_error"])
            except KeyError as e:
                raise ValueError(f"scores.json: metrics[{idx}].percentiles[{label}] missing key: {e}") from e
            except (TypeError, ValueError) as e:
                raise ValueError(f"scores.json: metrics[{idx}].percentiles[{label}] has invalid numeric values") from e
            percentiles.append(q)

        percentiles = sorted(set(percentiles))
        results.append(
            ScoreResult(
                metric=metric,
                percentiles=percentiles,
                sim=sim,
                real=real,
                pct_error=pct_error,
                verdict=verdict,
            )
        )

    return results


def write_summary_md(
    *,
    inputs: ReportInputs,
    results: list[ScoreResult],
    meta: dict,
    write_run_meta: bool = True,
    write_scores_json: bool = True,
    write_figures: bool = True,
) -> Path:
    """Write `summary.md` (and optional side artifacts) for a paper-fidelity run.

    Parameters
    ----------
    inputs
        Report inputs (paths and scenario name).
    results
        Scoring results.
    meta
        Run metadata written to `run_meta.json` when enabled.
    write_run_meta
        Whether to write `run_meta.json`.
    write_scores_json
        Whether to write `scores.json`.
    write_figures
        Whether to (re)generate SVG figures under `figs/`.
    """
    if not (write_run_meta or write_scores_json or write_figures) and not inputs.out_dir.exists():
        raise ValueError(f"Report directory does not exist: {inputs.out_dir}")
    inputs.out_dir.mkdir(parents=True, exist_ok=True)

    summary_md = inputs.out_dir / "summary.md"
    if write_scores_json:
        tables_dir = inputs.out_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = inputs.out_dir / "figs"
    if write_figures:
        figs_dir.mkdir(parents=True, exist_ok=True)

    if write_run_meta:
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
        cpu_overhead = profiling.get("cpu_overhead")

        lines.append("## Profiling")
        if root:
            lines.append(f"- root: `{root}`")
        if mode:
            lines.append(f"- mode: `{mode}`")
        if interpretation:
            lines.append(f"- interpretation: {interpretation}")
        if isinstance(cpu_overhead, dict):
            skip = cpu_overhead.get("skip_cpu_overhead_modeling")
            validation_mode = cpu_overhead.get("validation_mode")
            csv_path = cpu_overhead.get("cpu_overheads_csv")
            status = cpu_overhead.get("status")
            profiled = cpu_overhead.get("profiling_meta_cpu_overhead_profiled")
            error = cpu_overhead.get("error")
            warnings_list = cpu_overhead.get("warnings")

            lines.append("- cpu_overhead:")
            if skip is not None:
                lines.append(f"  - modeling: `{'disabled' if bool(skip) else 'enabled'}`")
            if validation_mode is not None:
                lines.append(f"  - validation: `{validation_mode}`")
            if csv_path is not None:
                lines.append(f"  - csv: `{csv_path}`")
            if status is not None:
                lines.append(f"  - status: `{status}`")
            if profiled is not None:
                lines.append(f"  - profiled: `{profiled}`")
            if error:
                lines.append(f"  - error: {error}")
            if isinstance(warnings_list, list) and warnings_list:
                lines.append("  - warnings:")
                for warning in warnings_list:
                    if isinstance(warning, str) and warning.strip():
                        lines.append(f"    - {warning}")
        lines.append("")

    paper_reference = meta.get("paper_reference")
    paper_values: dict[tuple[str, str], float] = {}
    paper_rows: list[dict] = []
    paper_requested = False
    paper_is_legacy = False
    paper_error = None
    if isinstance(paper_reference, dict):
        paper_is_legacy = "requested" not in paper_reference
        paper_requested = bool(paper_reference.get("requested") or False)
        paper_error = paper_reference.get("error")
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

    paper_requested = paper_requested or bool(paper_rows)

    if paper_requested:
        criteria = paper_reference.get("criteria") if isinstance(paper_reference, dict) else None
        workload_mode = paper_reference.get("workload_mode") if isinstance(paper_reference, dict) else None
        matched = paper_reference.get("matched") if isinstance(paper_reference, dict) else None
        load_frac = paper_reference.get("load_frac_of_capacity") if isinstance(paper_reference, dict) else None

        sources: list[str] = []
        for row in paper_rows:
            source_json = row.get("source_json")
            if isinstance(source_json, str) and source_json not in sources:
                sources.append(source_json)

        lines.append("## Paper Reference")
        if paper_is_legacy and paper_rows:
            first = paper_rows[0]
            model = first.get("model")
            trace = first.get("trace")
            series = first.get("series")
            if isinstance(model, str):
                lines.append(f"- model: `{model}`")
            if isinstance(trace, str):
                lines.append(f"- trace: `{trace}`")
            if isinstance(series, str):
                lines.append(f"- series: `{series}`")
        else:
            if workload_mode is not None:
                lines.append(f"- workload_mode: `{workload_mode}`")
            if matched is not None:
                lines.append(f"- matched: `{matched}`")
            if criteria and isinstance(criteria, dict):
                model = criteria.get("model")
                trace = criteria.get("trace")
                series = criteria.get("series")
                metric = criteria.get("metric")
                if isinstance(model, str):
                    lines.append(f"- model: `{model}`")
                if isinstance(trace, str):
                    lines.append(f"- trace: `{trace}`")
                if isinstance(series, str):
                    lines.append(f"- series: `{series}`")
                if isinstance(metric, str):
                    lines.append(f"- metric: `{metric}`")
            if load_frac is not None:
                lines.append(f"- load_frac_of_capacity: `{load_frac}`")
        if paper_error:
            lines.append(f"- error: `{paper_error}`")
        if sources:
            lines.append(f"- sources: {', '.join([f'`{s}`' for s in sources])}")
        if not paper_rows and not paper_is_legacy:
            lines.append("- rows: `0`")
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

    if write_scores_json:
        scores: dict[str, object] = {
            "schema_version": "v1",
            "scenario_name": inputs.scenario_name,
            "generated_at": _utcnow_iso(),
            "inputs": {"sim_csv": str(inputs.sim_csv), "real_csv": str(inputs.real_csv)},
            "metrics": [],
        }
        if isinstance(paper_reference, dict):
            scores["paper_reference"] = paper_reference
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
                        entry["sim_vs_paper_pct_error"] = float(
                            _percent_error(sim_value=float(r.sim[q]), ref_value=float(paper_value))
                        )
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
    figure_specs = [
        ("Static normalized latency", "request_execution_plus_preemption_time_normalized"),
        ("Dynamic normalized latency", "request_e2e_time_normalized"),
    ]
    if write_figures:
        try:
            sim_df = load_metrics_csv(inputs.sim_csv)
            real_df = load_metrics_csv(inputs.real_csv)

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
    else:
        found = False
        for title, metric in figure_specs:
            out_svg = figs_dir / f"{metric}_ecdf.svg"
            pct_svg = figs_dir / f"{metric}_percentiles.svg"
            if out_svg.exists():
                found = True
                lines.append(f"### {title}")
                lines.append(f"![ECDF: {metric}](figs/{out_svg.name})")
                if pct_svg.exists():
                    lines.append(f"![Percentiles: {metric}](figs/{pct_svg.name})")
                lines.append("")
        if not found:
            lines.append("- Figures not regenerated (no existing SVGs found).")
            lines.append("")

    if any(r.verdict in {"warn", "fail"} for r in results):
        lines.append("## Gap Diagnosis")
        for item in diagnose_gap(sim_csv=inputs.sim_csv, real_csv=inputs.real_csv, sim_meta=meta):
            lines.append(f"- {item}")
        lines.append("")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_md


def regenerate_summary_md_from_report_dir(
    report_dir: Path,
    *,
    include_paper_reference: bool = True,
) -> Path:
    """Regenerate `summary.md` for an existing report directory.

    This reads `run_meta.json` and `scores.json` from `report_dir` and rewrites
    `summary.md` in-place without modifying other artifacts (JSON or figures).

    Parameters
    ----------
    report_dir
        A `results/reports/.../paper_fidelity/<scenario>/` directory containing
        `run_meta.json` and `scores.json`.
    include_paper_reference
        Whether to include paper reference stats in the regenerated report, if
        present in `run_meta.json` and/or `scores.json`.

    Returns
    -------
    pathlib.Path
        Path to the regenerated `summary.md`.
    """
    report_dir = report_dir.expanduser()
    if not report_dir.exists():
        raise ValueError(f"Report directory does not exist: {report_dir}")

    meta_path = report_dir / "run_meta.json"
    scores_path = report_dir / "scores.json"
    if not meta_path.exists():
        raise ValueError(f"Missing run metadata: {meta_path}")
    if not scores_path.exists():
        raise ValueError(f"Missing scores JSON: {scores_path}")

    meta = read_json(meta_path)
    scores = read_json(scores_path)

    scenario_name = scores.get("scenario_name") or meta.get("scenario_name") or report_dir.name
    if not isinstance(scenario_name, str) or not scenario_name:
        raise ValueError(f"{scores_path}: missing scenario_name")

    inputs = scores.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"{scores_path}: missing inputs")
    sim_csv = inputs.get("sim_csv")
    real_csv = inputs.get("real_csv")
    if not isinstance(sim_csv, str) or not isinstance(real_csv, str):
        raise ValueError(f"{scores_path}: inputs.sim_csv and inputs.real_csv must be strings")

    if include_paper_reference:
        if "paper_reference" not in meta and "paper_reference" in scores:
            meta = {**meta, "paper_reference": scores["paper_reference"]}
    else:
        if "paper_reference" in meta:
            meta = {k: v for k, v in meta.items() if k != "paper_reference"}

    results = load_score_results_from_scores_json(scores)
    return write_summary_md(
        inputs=ReportInputs(
            scenario_name=scenario_name,
            sim_csv=Path(sim_csv),
            real_csv=Path(real_csv),
            out_dir=report_dir,
        ),
        results=results,
        meta=meta,
        write_run_meta=False,
        write_scores_json=False,
        write_figures=False,
    )
