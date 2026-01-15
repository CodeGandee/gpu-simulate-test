from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gpu_simulate_test.io import read_json, utcnow_iso, write_csv, write_json
from gpu_simulate_test.paper_fidelity.scoring import (
    PAPER_FIDELITY_REQUEST_METRICS_REQUIRED_COLUMNS,
    ScoreThresholds,
    load_metrics_csv,
    score_metric,
    validate_sim_vs_real_compatibility,
)
from gpu_simulate_test.vidur_cli.errors import UserFacingError
from gpu_simulate_test.vidur_ext.sim_runner import convert_vidur_request_metrics_to_paper_fidelity


@dataclass(frozen=True)
class ReportPaths:
    report_dir: Path
    summary_md: Path
    scores_json: Path
    inputs_dir: Path
    figs_dir: Path
    tables_dir: Path


def _fmt_pct(x: float) -> str:
    if x == float("inf"):
        return "inf"
    return f"{x * 100:.2f}%"


def _percent_error(*, sim_value: float, real_value: float) -> float:
    if real_value == 0.0:
        if sim_value == 0.0:
            return 0.0
        return float("inf")
    return abs(sim_value - real_value) / abs(real_value)


def _write_ecdf_svg(*, sim_values: np.ndarray, real_values: np.ndarray, metric: str, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required to write SVG plots; run inside the Pixi env.") from e

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
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required to write SVG plots; run inside the Pixi env.") from e

    labels = [f"p{int(q * 100)}" for q in percentiles]
    sim_vals = [float(sim[q]) for q in percentiles]
    real_vals = [float(real[q]) for q in percentiles]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=150)
    ax.bar(x - width / 2, real_vals, width, label="real")
    ax.bar(x + width / 2, sim_vals, width, label="sim")
    ax.set_title(f"Percentiles: {metric}")
    ax.set_ylabel("normalized latency")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _load_trace_meta(*, run_dir: Path) -> dict[str, Any]:
    trace_meta_path = run_dir / "trace" / "trace_meta.json"
    if not trace_meta_path.exists():
        raise UserFacingError(
            "Missing prerequisite: trace_meta.json not found under <run_dir>/trace/.",
            hint="Run `vidur-cli svr trace --run-dir <run_dir>` first.",
        )
    meta = read_json(trace_meta_path)
    return meta if isinstance(meta, dict) else {}


def _load_trace_csv_summary(*, run_dir: Path) -> dict[str, Any]:
    """Return a small summary of the canonical trace (counts + token maxima)."""
    from gpu_simulate_test.io import read_csv
    import pandas as pd  # type: ignore

    trace_csv = run_dir / "trace" / "trace.csv"
    if not trace_csv.exists():
        return {"trace_csv": str(trace_csv), "status": "missing"}

    df = read_csv(
        trace_csv,
        required_columns=["request_id", "arrival_time_ns", "num_prefill_tokens", "num_decode_tokens"],
        context="trace.csv",
    )
    num_requests = int(len(df))
    total_tokens = (
        pd.to_numeric(df["num_prefill_tokens"], errors="raise").astype(int)
        + pd.to_numeric(df["num_decode_tokens"], errors="raise").astype(int)
    )
    return {
        "trace_csv": str(trace_csv.resolve()),
        "status": "ok",
        "num_requests": num_requests,
        "max_prefill_tokens": int(pd.to_numeric(df["num_prefill_tokens"], errors="raise").astype(int).max())
        if num_requests
        else 0,
        "max_decode_tokens": int(pd.to_numeric(df["num_decode_tokens"], errors="raise").astype(int).max())
        if num_requests
        else 0,
        "max_total_tokens": int(total_tokens.max()) if num_requests else 0,
    }


def _read_arrival_kind(trace_meta: dict[str, Any]) -> str:
    arrival = trace_meta.get("arrival_schedule")
    if not isinstance(arrival, dict):
        return "unknown"
    kind = arrival.get("kind")
    return str(kind) if kind is not None else "unknown"


def _arrival_params(trace_meta: dict[str, Any]) -> dict[str, Any]:
    arrival = trace_meta.get("arrival_schedule")
    if not isinstance(arrival, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ["kind", "seed", "inter_arrival_ns", "poisson_rate_per_s"]:
        if key in arrival:
            out[key] = arrival.get(key)
    return out


def _detect_sim_pf_metrics_csv(*, sim_run_dir: Path) -> Path:
    # Preferred: materialized by `svr sim` under sim/paper_fidelity/request_metrics.csv
    candidate = sim_run_dir / "paper_fidelity" / "request_metrics.csv"
    if candidate.exists():
        return candidate

    # Fallback: derive from Vidur's raw outputs under sim/run_meta.json -> vidur_raw_dir.
    meta = read_json(sim_run_dir / "run_meta.json")
    raw_dir_value = meta.get("vidur_raw_dir") if isinstance(meta, dict) else None
    if not isinstance(raw_dir_value, str) or not raw_dir_value:
        raise UserFacingError(
            "sim/run_meta.json is missing vidur_raw_dir; cannot build paper-fidelity request_metrics.csv.",
            hint="Re-run `vidur-cli svr sim --run-dir <run_dir>`.",
        )
    raw_csv = Path(raw_dir_value).expanduser() / "request_metrics.csv"
    if not raw_csv.exists():
        raise UserFacingError(
            "Vidur raw request_metrics.csv is missing; cannot build paper-fidelity request_metrics.csv.",
            hint="Re-run `vidur-cli svr sim --run-dir <run_dir>`.",
            context={"raw_csv": str(raw_csv)},
        )
    df = convert_vidur_request_metrics_to_paper_fidelity(raw_csv)
    write_csv(candidate, df, required_columns=PAPER_FIDELITY_REQUEST_METRICS_REQUIRED_COLUMNS)
    return candidate


def _detect_real_pf_metrics_csv(*, real_run_dir: Path) -> Path:
    candidate = real_run_dir / "paper_fidelity" / "request_metrics.csv"
    if not candidate.exists():
        raise UserFacingError(
            "Missing prerequisite: Sarathi paper-fidelity request metrics are missing.",
            hint="Re-run `vidur-cli svr real --run-dir <run_dir>` using backend=sarathi.",
            context={"expected_path": str(candidate)},
        )
    return candidate


def write_paper_fidelity_style_report(
    *,
    run_dir: Path,
    report_dir: Path,
    sim_run_dir: Path,
    real_run_dir: Path,
    profiling_root: Path,
    include_cpu_overhead: bool,
) -> Path:
    run_dir = run_dir.expanduser().resolve()
    report_dir = report_dir.expanduser().resolve()
    sim_run_dir = sim_run_dir.expanduser().resolve()
    real_run_dir = real_run_dir.expanduser().resolve()

    paths = ReportPaths(
        report_dir=report_dir,
        summary_md=report_dir / "summary.md",
        scores_json=report_dir / "scores.json",
        inputs_dir=report_dir / "inputs",
        figs_dir=report_dir / "figs",
        tables_dir=report_dir / "tables",
    )

    trace_meta = _load_trace_meta(run_dir=run_dir)
    arrival_kind = _read_arrival_kind(trace_meta)
    arrival_params = _arrival_params(trace_meta)
    trace_summary = _load_trace_csv_summary(run_dir=run_dir)

    sim_pf_csv_src = _detect_sim_pf_metrics_csv(sim_run_dir=sim_run_dir)
    real_pf_csv_src = _detect_real_pf_metrics_csv(real_run_dir=real_run_dir)

    paths.report_dir.mkdir(parents=True, exist_ok=True)
    paths.inputs_dir.mkdir(parents=True, exist_ok=True)
    paths.figs_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)

    sim_pf_csv = paths.inputs_dir / "sim_request_metrics.csv"
    real_pf_csv = paths.inputs_dir / "real_request_metrics.csv"
    # Use stable copies inside the report directory for reproducibility.
    sim_pf_csv.write_bytes(sim_pf_csv_src.read_bytes())
    real_pf_csv.write_bytes(real_pf_csv_src.read_bytes())

    sim_df = load_metrics_csv(sim_pf_csv)
    real_df = load_metrics_csv(real_pf_csv)
    validate_sim_vs_real_compatibility(sim_df=sim_df, real_df=real_df)

    percentiles = [0.5, 0.95]
    thresholds = ScoreThresholds()
    metrics = [
        "request_execution_plus_preemption_time_normalized",
        "request_e2e_time_normalized",
        "prefill_time_execution_plus_preemption_normalized",
        "decode_time_execution_plus_preemption_normalized",
    ]

    results = [score_metric(sim_df=sim_df, real_df=real_df, metric=m, percentiles=percentiles, thresholds=thresholds) for m in metrics]

    # scores.json (similar to paper-fidelity, but report rendering omits verdict).
    scores_payload: dict[str, Any] = {
        "schema_version": "v1",
        "generated_at": utcnow_iso(),
        "percentiles": [f"p{int(q * 100)}" for q in percentiles],
        "metrics": [],
    }
    for r in results:
        entry: dict[str, Any] = {"metric": r.metric, "percentiles": {}}
        for q in r.percentiles:
            label = f"p{int(q * 100)}"
            entry["percentiles"][label] = {
                "sim": float(r.sim[q]),
                "real": float(r.real[q]),
                "pct_error": float(r.pct_error[q]),
            }
        scores_payload["metrics"].append(entry)
    write_json(paths.scores_json, scores_payload)

    # Figures for every metric in the score table.
    for r in results:
        metric = r.metric
        sim_values = np.asarray(sim_df[metric], dtype=float)
        real_values = np.asarray(real_df[metric], dtype=float)
        sim_values = sim_values[np.isfinite(sim_values)]
        real_values = real_values[np.isfinite(real_values)]
        if len(sim_values) == 0 or len(real_values) == 0:
            continue

        ecdf_svg = paths.figs_dir / f"{metric}_ecdf.svg"
        pct_svg = paths.figs_dir / f"{metric}_percentiles.svg"
        _write_ecdf_svg(sim_values=sim_values, real_values=real_values, metric=metric, out_path=ecdf_svg)
        _write_percentiles_svg(metric=metric, percentiles=r.percentiles, sim=r.sim, real=r.real, out_path=pct_svg)

    # Config parity (critical knobs only).
    sim_meta = read_json(sim_run_dir / "run_meta.json")
    real_meta = read_json(real_run_dir / "run_meta.json")
    sim_meta = sim_meta if isinstance(sim_meta, dict) else {}
    real_meta = real_meta if isinstance(real_meta, dict) else {}

    model_id = sim_meta.get("model_id") or sim_meta.get("model") or real_meta.get("model") or "unknown"
    sim_model_ref = sim_meta.get("model_ref") or "unknown"
    real_model_ref = real_meta.get("model_ref") or "unknown"

    sim_sched = sim_meta.get("scheduler") if isinstance(sim_meta.get("scheduler"), dict) else {}
    real_sched = real_meta.get("scheduler") if isinstance(real_meta.get("scheduler"), dict) else {}
    sim_cpu = sim_meta.get("cpu_overhead") if isinstance(sim_meta.get("cpu_overhead"), dict) else {}

    sim_chunk = sim_sched.get("chunk_size")
    sim_batch = sim_sched.get("batch_size_cap")
    sim_block = sim_sched.get("block_size")
    sim_watermark = sim_sched.get("watermark_blocks_fraction")
    sim_max_tokens = sim_meta.get("max_tokens")
    sim_tp = sim_meta.get("tensor_parallel_size")
    sim_pp = sim_meta.get("num_pipeline_stages")
    sim_skip_cpu = sim_cpu.get("skip_cpu_overhead_modeling")

    real_chunk = real_sched.get("chunk_size")
    real_batch = real_sched.get("max_num_seqs")
    real_max_tokens = real_meta.get("max_tokens")
    real_ignore_eos = real_meta.get("ignore_eos")
    real_parallel = real_meta.get("parallel") if isinstance(real_meta.get("parallel"), dict) else {}
    real_tp = real_parallel.get("tensor_parallel_size")
    real_pp = real_parallel.get("pipeline_parallel_size")

    def _status(a: object, b: object) -> str:
        if a in {None, "unknown"} or b in {None, "unknown"}:
            return "unknown"
        return "match" if a == b else "MISMATCH"

    profiling_root = profiling_root.expanduser().resolve()
    cpu_overheads_csv = (
        profiling_root
        / "data"
        / "profiling"
        / "cpu_overhead"
        / "a100_pairwise_nvlink"
        / str(model_id)
        / "cpu_overheads.csv"
    )
    cpu_overheads_status = "missing"
    if include_cpu_overhead:
        cpu_overheads_status = "ok" if cpu_overheads_csv.exists() else "missing"
    else:
        cpu_overheads_status = "skipped"

    # Summary.md
    lines: list[str] = []
    lines.append(f"# Sim-vs-Real Report: {run_dir.name}")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- sim: `{sim_pf_csv}`")
    lines.append(f"- real: `{real_pf_csv}`")
    lines.append("")

    lines.append("## Profiling")
    lines.append(f"- root: `{profiling_root}`")
    lines.append(f"- cpu_overhead:")
    lines.append(f"  - modeling: `{'enabled' if include_cpu_overhead else 'disabled'}`")
    lines.append(f"  - csv: `{cpu_overheads_csv}`")
    lines.append(f"  - status: `{cpu_overheads_status}`")
    lines.append("")

    lines.append("## Config (apple-to-apple)")
    lines.append(f"- model_id: `{model_id}`")
    lines.append(f"- model_ref (sim): `{sim_model_ref}`")
    lines.append(f"- model_ref (real): `{real_model_ref}`")
    lines.append(f"- arrival_kind: `{arrival_kind}`")
    if arrival_params:
        params_str = ", ".join([f"{k}={arrival_params[k]}" for k in sorted(arrival_params) if k != "kind"])
        if params_str:
            lines.append(f"- arrival_params: `{params_str}`")
    if trace_summary.get("status") == "ok":
        lines.append(
            f"- trace: num_requests=`{trace_summary.get('num_requests')}` "
            f"max_total_tokens=`{trace_summary.get('max_total_tokens')}`"
        )
    lines.append(f"- max_tokens: sim=`{sim_max_tokens}` real=`{real_max_tokens}` ({_status(sim_max_tokens, real_max_tokens)})")
    lines.append(f"- ignore_eos (real): `{real_ignore_eos}`")
    lines.append(f"- tensor_parallel_size: sim=`{sim_tp}` real=`{real_tp}` ({_status(sim_tp, real_tp)})")
    lines.append(f"- pipeline_parallel_size: sim=`{sim_pp}` real=`{real_pp}` ({_status(sim_pp, real_pp)})")
    lines.append(f"- chunk_size: sim=`{sim_chunk}` real=`{real_chunk}` ({_status(sim_chunk, real_chunk)})")
    lines.append(f"- batch_size: sim=`{sim_batch}` real=`{real_batch}` ({_status(sim_batch, real_batch)})")
    lines.append(f"- block_size (sim): `{sim_block}`")
    lines.append(f"- watermark_blocks_fraction (sim): `{sim_watermark}`")
    lines.append(f"- cpu_overhead_modeling (sim): `{'disabled' if bool(sim_skip_cpu) else 'enabled'}`")
    if sim_max_tokens is None or sim_tp is None or sim_chunk is None:
        lines.append("- WARNING: sim/run_meta.json is missing parity-critical fields; rerun `vidur-cli svr sim` to populate.")
    if include_cpu_overhead and bool(sim_skip_cpu):
        lines.append("  - WARNING: sim reports skip_cpu_overhead_modeling=true but profile include_cpu_overhead=true.")
    if not include_cpu_overhead and not bool(sim_skip_cpu):
        lines.append("  - WARNING: sim reports skip_cpu_overhead_modeling=false but profile include_cpu_overhead=false.")
    lines.append("")

    lines.append("## Scores")
    lines.append("| Metric | Percentile | Sim | Real | Percent error |")
    lines.append("|--------|------------|-----|------|---------------|")
    for r in results:
        for q in r.percentiles:
            label = f"p{int(q * 100)}"
            sim_val = float(r.sim[q])
            real_val = float(r.real[q])
            pct = _fmt_pct(_percent_error(sim_value=sim_val, real_value=real_val))
            lines.append(f"| {r.metric} | {label} | {sim_val:.6g} | {real_val:.6g} | {pct} |")
    lines.append("")

    lines.append("## Figures")
    for r in results:
        metric = r.metric
        ecdf_svg = paths.figs_dir / f"{metric}_ecdf.svg"
        pct_svg = paths.figs_dir / f"{metric}_percentiles.svg"
        if not ecdf_svg.exists():
            continue
        lines.append(f"### Metric: {metric}")
        lines.append(f"![ECDF: {metric}](figs/{ecdf_svg.name})")
        if pct_svg.exists():
            lines.append(f"![Percentiles: {metric}](figs/{pct_svg.name})")
        lines.append("")

    write_json(
        report_dir / "run_meta.json",
        {
            "schema_version": "v1",
            "generated_at": utcnow_iso(),
            "run_dir": str(run_dir),
            "sim_run_dir": str(sim_run_dir),
            "real_run_dir": str(real_run_dir),
            "inputs": {
                "sim_request_metrics_csv": str(sim_pf_csv.resolve()),
                "real_request_metrics_csv": str(real_pf_csv.resolve()),
            },
            "profiling": {
                "root": str(profiling_root),
                "cpu_overhead": {
                    "include_cpu_overhead": bool(include_cpu_overhead),
                    "cpu_overheads_csv": str(cpu_overheads_csv),
                    "status": cpu_overheads_status,
                },
            },
            "config": {
                "model_id": model_id,
                "sim_model_ref": sim_model_ref,
                "real_model_ref": real_model_ref,
                "arrival_kind": arrival_kind,
                "arrival_params": arrival_params,
                "trace": trace_summary,
                "sim": {
                    "max_tokens": sim_max_tokens,
                    "tensor_parallel_size": sim_tp,
                    "num_pipeline_stages": sim_pp,
                    "scheduler": sim_sched,
                    "cpu_overhead": sim_cpu,
                },
                "real": {
                    "max_tokens": real_max_tokens,
                    "tensor_parallel_size": real_tp,
                    "pipeline_parallel_size": real_pp,
                    "ignore_eos": real_ignore_eos,
                    "scheduler": real_sched,
                },
            },
        },
    )

    paths.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths.summary_md.resolve()
