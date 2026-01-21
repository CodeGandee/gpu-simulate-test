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


def _extract_override_value(overrides: list[object], *, key: str) -> str | None:
    for item in overrides:
        if not isinstance(item, str):
            continue
        if "=" not in item:
            continue
        raw_key, value = item.split("=", 1)
        normalized_key = raw_key.lstrip("+~")
        if normalized_key == key:
            return value
    return None


def _infer_mlp_profile_method_from_staging(*, run_dir: Path) -> str | None:
    """Infer the MLP profiling method from the latest staging config, if present."""
    staging_root = run_dir / "profile" / "_staging" / "mlp"
    if not staging_root.exists():
        return None

    candidates = [p for p in staging_root.glob("*") if p.is_dir()]
    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    config_yaml = latest / "config.yaml"
    if not config_yaml.exists():
        return None

    for line in config_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("profile_method:"):
            continue
        _, value = stripped.split(":", 1)
        method = value.strip().strip("\"'").strip()
        return method or None
    return None


def _load_profile_mlp_selection(*, run_dir: Path) -> dict[str, Any] | None:
    """Best-effort extraction of MLP profiling selections for the final report.

    The profiling method selection is critical for reproducibility. Prefer the structured
    `artifacts.profile.mlp` payload written by `svr profile`, but fall back to parsing
    `artifacts.profile.overrides` for older runs.
    """
    run_state_path = run_dir / "run_state.json"
    if not run_state_path.exists():
        return None

    state = read_json(run_state_path)
    if not isinstance(state, dict):
        return None
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    profile = artifacts.get("profile")
    if not isinstance(profile, dict):
        return None

    mlp = profile.get("mlp")
    if isinstance(mlp, dict) and mlp:
        return dict(mlp)

    overrides = profile.get("overrides")
    if not isinstance(overrides, list):
        return None

    profile_method = _extract_override_value(overrides, key="profiling.mlp.profile_method")
    if profile_method is None:
        inferred = _infer_mlp_profile_method_from_staging(run_dir=run_dir)
        if inferred is None:
            return None
        return {
            "requested_profile_method": str(inferred),
            "profile_method": str(inferred),
        }

    out: dict[str, Any] = {
        "requested_profile_method": str(profile_method),
        "profile_method": str(profile_method),
    }
    validation_mode = _extract_override_value(overrides, key="profiling.mlp.validation.mode")
    if validation_mode is not None:
        out["validation_mode"] = str(validation_mode)
    nan_policy = _extract_override_value(overrides, key="profiling.mlp.validation.nan_policy")
    if nan_policy is not None:
        out["nan_policy"] = str(nan_policy)
    small_input_threshold = _extract_override_value(
        overrides, key="profiling.mlp.validation.small_input_threshold"
    )
    if small_input_threshold is not None:
        out["small_input_threshold"] = int(small_input_threshold)
    zero_heavy_limit = _extract_override_value(overrides, key="profiling.mlp.validation.zero_heavy_limit")
    if zero_heavy_limit is not None:
        out["zero_heavy_limit"] = float(zero_heavy_limit)
    fallback_enabled = _extract_override_value(overrides, key="profiling.mlp.fallback.enabled")
    if fallback_enabled is not None:
        out["fallback_enabled"] = str(fallback_enabled).lower().strip() in {"1", "true", "yes", "y", "on"}
    fallback_method = _extract_override_value(overrides, key="profiling.mlp.fallback.method")
    if fallback_method is not None:
        out["fallback_method"] = str(fallback_method)
    return out


def _load_profile_resolved(*, run_dir: Path) -> dict[str, Any] | None:
    """Load resolved profiling settings recorded by `svr profile` (best effort)."""
    run_state_path = run_dir / "run_state.json"
    if not run_state_path.exists():
        return None

    state = read_json(run_state_path)
    if not isinstance(state, dict):
        return None
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    profile = artifacts.get("profile")
    if not isinstance(profile, dict):
        return None

    resolved = profile.get("resolved")
    if isinstance(resolved, dict) and resolved:
        return dict(resolved)
    return None


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


def _write_sim_request_metrics_csv(*, sim_run_dir: Path, out_csv: Path) -> None:
    """Write paper-fidelity-style request metrics for sim into the report inputs directory.

    Note: we intentionally avoid writing any derived artifacts under `<run_dir>/sim/` so users
    don't confuse "paper_fidelity" the metric schema with the separate `paper-fidelity` pipeline.
    """
    # Back-compat: if older runs materialized this file already, just copy it.
    legacy = sim_run_dir / "paper_fidelity" / "request_metrics.csv"
    if legacy.exists():
        out_csv.write_bytes(legacy.read_bytes())
        return

    meta = read_json(sim_run_dir / "run_meta.json")
    raw_dir_value = meta.get("vidur_raw_dir") if isinstance(meta, dict) else None
    if not isinstance(raw_dir_value, str) or not raw_dir_value:
        raise UserFacingError(
            "sim/run_meta.json is missing vidur_raw_dir; cannot build paper-fidelity request metrics.",
            hint="Re-run `vidur-cli svr sim --run-dir <run_dir>`.",
        )

    raw_csv = Path(raw_dir_value).expanduser() / "request_metrics.csv"
    if not raw_csv.exists():
        raise UserFacingError(
            "Vidur raw request_metrics.csv is missing; cannot build paper-fidelity request metrics.",
            hint="Re-run `vidur-cli svr sim --run-dir <run_dir>`.",
            context={"raw_csv": str(raw_csv)},
        )

    df = convert_vidur_request_metrics_to_paper_fidelity(raw_csv)
    write_csv(out_csv, df, required_columns=PAPER_FIDELITY_REQUEST_METRICS_REQUIRED_COLUMNS)


def _write_real_request_metrics_csv(*, real_run_dir: Path, out_csv: Path) -> None:
    """Write paper-fidelity-style request metrics for real into the report inputs directory."""
    # Back-compat: if older runs materialized this file already, just copy it.
    legacy = real_run_dir / "paper_fidelity" / "request_metrics.csv"
    if legacy.exists():
        out_csv.write_bytes(legacy.read_bytes())
        return

    sarathi_sequence_metrics_csv = real_run_dir / "sarathi" / "replica_0" / "sequence_metrics.csv"
    if not sarathi_sequence_metrics_csv.exists():
        raise UserFacingError(
            "Missing prerequisite: Sarathi sequence_metrics.csv is missing.",
            hint="Re-run `vidur-cli svr real --run-dir <run_dir>` using backend=sarathi.",
            context={"expected_path": str(sarathi_sequence_metrics_csv)},
        )

    from gpu_simulate_test.real_bench.backends.sarathi_paper_fidelity_backend import (
        convert_sequence_metrics_to_request_metrics,
    )

    df = convert_sequence_metrics_to_request_metrics(sarathi_sequence_metrics_csv)
    write_csv(out_csv, df, required_columns=PAPER_FIDELITY_REQUEST_METRICS_REQUIRED_COLUMNS)


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

    paths.report_dir.mkdir(parents=True, exist_ok=True)
    paths.inputs_dir.mkdir(parents=True, exist_ok=True)
    paths.figs_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)

    sim_pf_csv = paths.inputs_dir / "sim_request_metrics.csv"
    real_pf_csv = paths.inputs_dir / "real_request_metrics.csv"
    # Materialize stable inputs inside the report directory for reproducibility.
    _write_sim_request_metrics_csv(sim_run_dir=sim_run_dir, out_csv=sim_pf_csv)
    _write_real_request_metrics_csv(real_run_dir=real_run_dir, out_csv=real_pf_csv)

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

    def _fmt_bool(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    profiling_root = profiling_root.expanduser().resolve()
    profile_resolved = _load_profile_resolved(run_dir=run_dir)
    resolved_network_device = (
        profile_resolved.get("network_device") if isinstance(profile_resolved, dict) else None
    )
    network_device = str(resolved_network_device) if resolved_network_device else "a100_pairwise_nvlink"
    cpu_overheads_csv = (
        profiling_root
        / "data"
        / "profiling"
        / "cpu_overhead"
        / network_device
        / str(model_id)
        / "cpu_overheads.csv"
    )
    cpu_overheads_status = "missing"
    if include_cpu_overhead:
        cpu_overheads_status = "ok" if cpu_overheads_csv.exists() else "missing"
    else:
        cpu_overheads_status = "skipped"

    mlp_selection = _load_profile_mlp_selection(run_dir=run_dir)
    sim_mlp_validation = sim_meta.get("mlp_validation") if isinstance(sim_meta.get("mlp_validation"), dict) else {}
    sim_mlp_nan_drop = sim_meta.get("mlp_nan_drop") if isinstance(sim_meta.get("mlp_nan_drop"), dict) else {}
    sim_mlp_nan_fill_zero = (
        sim_meta.get("mlp_nan_fill_zero") if isinstance(sim_meta.get("mlp_nan_fill_zero"), dict) else {}
    )

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
    if profile_resolved is not None:
        lines.append("- settings:")
        for key in ["num_gpus", "tensor_parallel_size", "max_tokens", "include_network"]:
            if key in profile_resolved:
                lines.append(f"  - {key}: `{profile_resolved.get(key)}`")
    lines.append(f"- cpu_overhead:")
    lines.append(f"  - modeling: `{'enabled' if include_cpu_overhead else 'disabled'}`")
    lines.append(f"  - network_device: `{network_device}`")
    if (
        profile_resolved is not None
        and isinstance(profile_resolved.get("cpu_overhead"), dict)
    ):
        cpu_cfg = profile_resolved["cpu_overhead"]
        if "max_batch_size" in cpu_cfg:
            lines.append(f"  - max_batch_size: `{cpu_cfg.get('max_batch_size')}`")
        if "validation" in cpu_cfg:
            lines.append(f"  - validation: `{cpu_cfg.get('validation')}`")
    lines.append(f"  - csv: `{cpu_overheads_csv}`")
    lines.append(f"  - status: `{cpu_overheads_status}`")
    if (
        profile_resolved is not None
        and isinstance(profile_resolved.get("attention"), dict)
    ):
        attn_cfg = profile_resolved["attention"]
        parts: list[str] = []
        for key in ["profile_mode", "backend", "block_size", "min_batch_size", "max_batch_size"]:
            if key in attn_cfg and attn_cfg.get(key) is not None:
                parts.append(f"{key}={attn_cfg.get(key)}")
        if parts:
            lines.append(f"- attention: `{' '.join(parts)}`")
    lines.append("- mlp:")
    if mlp_selection is None:
        lines.append("  - profile_method: `unknown`")
        lines.append("  - WARNING: MLP profiling selection was not recorded; re-run `vidur-cli svr profile`.")
    else:
        requested_method = mlp_selection.get("requested_profile_method")
        profile_method = mlp_selection.get("profile_method") or requested_method or "unknown"
        if requested_method is not None and requested_method != profile_method:
            lines.append(f"  - profile_method: `{profile_method}` (fallback from `{requested_method}`)")
        else:
            lines.append(f"  - profile_method: `{profile_method}`")

        vidur_profile_method = mlp_selection.get("vidur_profile_method")
        if isinstance(vidur_profile_method, str) and vidur_profile_method:
            lines.append(f"  - vidur_profile_method: `{vidur_profile_method}`")

        validation_mode = mlp_selection.get("validation_mode")
        nan_policy = mlp_selection.get("nan_policy")
        small_input_threshold = mlp_selection.get("small_input_threshold")
        zero_heavy_limit = mlp_selection.get("zero_heavy_limit")
        if (
            validation_mode is not None
            or nan_policy is not None
            or small_input_threshold is not None
            or zero_heavy_limit is not None
        ):
            parts: list[str] = []
            if validation_mode is not None:
                parts.append(f"mode={validation_mode}")
            if nan_policy is not None:
                parts.append(f"nan_policy={nan_policy}")
            if small_input_threshold is not None:
                parts.append(f"small_input_threshold={small_input_threshold}")
            if zero_heavy_limit is not None:
                parts.append(f"zero_heavy_limit={zero_heavy_limit}")
            lines.append(f"  - validation: `{' '.join(parts)}`")

        fallback_enabled = mlp_selection.get("fallback_enabled")
        fallback_method = mlp_selection.get("fallback_method")
        fallback_used = mlp_selection.get("fallback_used")
        parts = []
        if fallback_enabled is not None:
            parts.append(f"enabled={_fmt_bool(fallback_enabled)}")
        if fallback_method is not None:
            parts.append(f"method={fallback_method}")
        if fallback_used is not None:
            parts.append(f"used={_fmt_bool(fallback_used)}")
        if parts:
            lines.append(f"  - fallback: `{' '.join(parts)}`")

    lines.append("- mlp_consumer:")
    if sim_mlp_validation:
        mode = sim_mlp_validation.get("mode") or "unknown"
        nan_policy = sim_mlp_validation.get("nan_policy") or "unknown"
        effective = sim_mlp_validation.get("nan_policy_effective") or "unknown"
        lines.append(f"  - nan_policy: `{effective}` (nan_policy=`{nan_policy}` mode=`{mode}`)")
    else:
        lines.append("  - nan_policy: `unknown`")

    if sim_mlp_nan_drop and bool(sim_mlp_nan_drop.get("enabled")):
        per_model = sim_mlp_nan_drop.get("per_model") if isinstance(sim_mlp_nan_drop.get("per_model"), dict) else {}
        dropped_total = 0
        models_with_drops = 0
        for stats in per_model.values():
            if not isinstance(stats, dict):
                continue
            dropped = int(stats.get("rows_dropped") or 0)
            dropped_total += dropped
            if dropped > 0:
                models_with_drops += 1
        lines.append(
            f"  - nan_drop: `enabled` models_with_drops=`{models_with_drops}` rows_dropped_total=`{dropped_total}`"
        )
    else:
        lines.append("  - nan_drop: `disabled`")

    if sim_mlp_nan_fill_zero and bool(sim_mlp_nan_fill_zero.get("enabled")):
        per_model = (
            sim_mlp_nan_fill_zero.get("per_model")
            if isinstance(sim_mlp_nan_fill_zero.get("per_model"), dict)
            else {}
        )
        filled_total = 0
        models_with_fills = 0
        for stats in per_model.values():
            if not isinstance(stats, dict):
                continue
            filled = int(stats.get("cells_filled") or 0)
            filled_total += filled
            if filled > 0:
                models_with_fills += 1
        lines.append(
            f"  - nan_fill_zero: `enabled` models_with_fills=`{models_with_fills}` cells_filled_total=`{filled_total}`"
        )
    else:
        lines.append("  - nan_fill_zero: `disabled`")
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
                "mlp": mlp_selection,
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
