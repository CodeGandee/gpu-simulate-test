"""
CLI entrypoint for the paper-fidelity reproduction workflow.

Commands
--------
repro
    End-to-end reproduction (trace → sim → capacity/real → score → report).
trace
    Generate/validate a canonical trace for a scenario/workload.
profile
    Generate a host profiling root for Vidur simulation.
score
    Score existing sim vs real metrics and write a report.
report
    Regenerate `summary.md` from an existing report directory (no rerun).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import hydra
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf

from gpu_simulate_test.config import register_omegaconf_resolvers
from gpu_simulate_test.io import build_env_snapshot, get_git_info, stable_id, utcnow_iso, write_json
from gpu_simulate_test.paper_fidelity.capacity import CapacityCriterion, discover_capacity, write_capacity_json
from gpu_simulate_test.paper_fidelity.failure_record import (
    build_failure_record,
    categorize_blocker,
    write_failure_record,
)
from gpu_simulate_test.paper_fidelity.paper_reference import maybe_load_paper_reference_rows_from_cfg
from gpu_simulate_test.paper_fidelity.paths import PaperFidelityPaths
from gpu_simulate_test.paper_fidelity.profiling import PaperFidelityProfilingError, run_paper_fidelity_profiling
from gpu_simulate_test.paper_fidelity.report import ReportInputs, regenerate_summary_md_from_report_dir, write_summary_md
from gpu_simulate_test.paper_fidelity.scoring import (
    ScoreThresholds,
    load_metrics_csv,
    score_metric,
    validate_sim_vs_real_compatibility,
)
from gpu_simulate_test.paper_fidelity.validation import (
    MissingTraceSourceError,
    ScenarioPreflightError,
    preflight_repro,
    preflight_trace,
)
from gpu_simulate_test.paper_fidelity.traces import (
    TraceSpec,
    add_poisson_arrivals,
    apply_trace_subset,
    legacy_workload_dir_to_trace,
    make_static,
    processed_lengths_csv_to_trace,
    read_trace_csv,
    validate_trace,
)
from gpu_simulate_test.real_bench.backends.sarathi_paper_fidelity_backend import (
    SarathiPaperFidelityInputs,
    run_sarathi_paper_fidelity,
)
from gpu_simulate_test.vidur_ext.sim_runner import VidurPaperFidelitySimInputs, run_vidur_paper_fidelity_sim


register_omegaconf_resolvers()


PaperFidelityWorkload = Literal["static", "dynamic"]


class PaperFidelityReproError(RuntimeError):
    def __init__(self, message: str, *, failure_record_path: Path) -> None:
        super().__init__(message)
        self.failure_record_path = failure_record_path


def main(argv: list[str] | None = None) -> None:
    """Dispatch `paper-fidelity` subcommands (argparse wrapper around Hydra apps)."""
    parser = argparse.ArgumentParser(prog="paper-fidelity")
    sub = parser.add_subparsers(dest="cmd", required=True)

    repro = sub.add_parser("repro")
    repro.add_argument("--scenario", required=True)
    repro.add_argument("--workload", choices=["static", "dynamic"], required=True)
    repro.add_argument("--scale", choices=["small", "medium", "full"], default=None)

    trace = sub.add_parser("trace")
    trace.add_argument("--scenario", required=True)
    trace.add_argument("--workload", choices=["static", "dynamic"], required=True)
    trace.add_argument("--scale", choices=["small", "medium", "full"], default=None)

    profile = sub.add_parser("profile")
    profile.add_argument("--scenario", required=True)
    profile.add_argument("--include-cpu-overhead", action="store_true")

    score = sub.add_parser("score")
    score.add_argument("--sim", required=True)
    score.add_argument("--real", required=True)

    report = sub.add_parser("report")
    report.add_argument("--dir", required=True)
    report.add_argument("--paper-reference", choices=["include", "omit"], default="include")

    args, hydra_overrides = parser.parse_known_args(argv)
    prog = sys.argv[0]

    if args.cmd == "repro":
        sys.argv = [prog, f"scenario={args.scenario}", f"workload={args.workload}", *hydra_overrides]
        if args.scale is not None:
            sys.argv.append(f"scale={args.scale}")
        _repro_main()
    elif args.cmd == "trace":
        sys.argv = [prog, f"scenario={args.scenario}", f"workload={args.workload}", *hydra_overrides]
        if args.scale is not None:
            sys.argv.append(f"scale={args.scale}")
        _trace_main()
    elif args.cmd == "profile":
        sys.argv = [prog, f"scenario={args.scenario}", *hydra_overrides]
        if args.include_cpu_overhead:
            sys.argv.append("profiling.include_cpu_overhead=true")
        _profile_main()
    elif args.cmd == "score":
        sim_csv = str(Path(args.sim).expanduser())
        real_csv = str(Path(args.real).expanduser())
        sys.argv = [prog, f"inputs.sim_csv={sim_csv}", f"inputs.real_csv={real_csv}", *hydra_overrides]
        _score_main()
    elif args.cmd == "report":
        if hydra_overrides:
            raise ValueError(f"`paper-fidelity report` does not accept Hydra overrides (got {hydra_overrides})")
        include_paper = str(args.paper_reference) == "include"
        summary_md = regenerate_summary_md_from_report_dir(
            Path(args.dir).expanduser(),
            include_paper_reference=include_paper,
        )
        print(str(summary_md))
    else:  # pragma: no cover
        raise ValueError(f"Unhandled cmd: {args.cmd}")


def _utc_date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _infer_scenario_name_from_metrics_csv(path: Path) -> str | None:
    parts = list(path.resolve().parts)
    try:
        pf_idx = parts.index("paper_fidelity")
    except ValueError:
        return None

    if pf_idx + 2 < len(parts) and parts[pf_idx + 1] == "runs":
        return parts[pf_idx + 2]
    return None


def _infer_scenario_name(*, sim_csv: Path, real_csv: Path) -> str:
    sim = _infer_scenario_name_from_metrics_csv(sim_csv)
    real = _infer_scenario_name_from_metrics_csv(real_csv)
    if sim and real and sim == real:
        return sim
    return sim or real or stable_id([str(sim_csv), str(real_csv)], prefix="adhoc", length=12)


def _inspect_cpu_overhead_inputs(
    *,
    profiling_root: Path,
    model_id: str,
    network_device: str,
    skip_cpu_overhead_modeling: bool,
    validation_mode: str,
    expected_tensor_parallel_size: int,
) -> dict[str, object]:
    cpu_overheads_csv = (
        profiling_root
        / "data"
        / "profiling"
        / "cpu_overhead"
        / str(network_device)
        / str(model_id)
        / "cpu_overheads.csv"
    )

    info: dict[str, object] = {
        "skip_cpu_overhead_modeling": bool(skip_cpu_overhead_modeling),
        "validation_mode": str(validation_mode),
        "cpu_overheads_csv": str(cpu_overheads_csv),
        "cpu_overheads_exists": bool(cpu_overheads_csv.exists()),
        "status": "disabled" if skip_cpu_overhead_modeling else "unknown",
        "profiling_meta_cpu_overhead_profiled": None,
        "warnings": [],
        "error": None,
    }

    meta_path = profiling_root / "profiling_meta.json"
    if meta_path.exists():
        try:
            import json

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = None
        if isinstance(meta, dict):
            profiled = None
            profiling_outputs = meta.get("profiling_outputs")
            vidur_profile_result = meta.get("vidur_profile_result")
            if isinstance(profiling_outputs, dict) and "cpu_overhead_profiled" in profiling_outputs:
                profiled = profiling_outputs.get("cpu_overhead_profiled")
            elif isinstance(vidur_profile_result, dict) and "cpu_overhead_profiled" in vidur_profile_result:
                profiled = vidur_profile_result.get("cpu_overhead_profiled")
            if isinstance(profiled, bool):
                info["profiling_meta_cpu_overhead_profiled"] = profiled

    if skip_cpu_overhead_modeling:
        return info

    if not cpu_overheads_csv.exists():
        info["status"] = "missing"
        info["error"] = f"Missing cpu_overheads.csv at {cpu_overheads_csv}"
        return info

    from gpu_simulate_test.vidur_ext.cpu_overhead_validation import validate_cpu_overheads_csv

    try:
        result = validate_cpu_overheads_csv(
            cpu_overheads_csv,
            mode=str(validation_mode).lower().strip() or "strict",  # type: ignore[arg-type]
            expected_model_id=str(model_id),
            expected_tensor_parallel_degree=int(expected_tensor_parallel_size),
        )
        info["validation_result"] = result.as_jsonable()
        info["status"] = "placeholder" if result.placeholder_like else "ok"
        if result.warnings:
            info["warnings"] = list(result.warnings)
        profiled = info.get("profiling_meta_cpu_overhead_profiled")
        if profiled is False:
            warnings_list = list(info.get("warnings") or [])
            warnings_list.append(
                "profiling_meta.json reports cpu_overhead_profiled=false; CPU overhead inputs may be unprofiled."
            )
            info["warnings"] = warnings_list
    except Exception as e:
        info["status"] = "error"
        info["error"] = str(e)

    return info


def _trace_subset_from_cfg(cfg: DictConfig) -> tuple[str, object, object, list[object] | None]:
    kind = str(OmegaConf.select(cfg, "trace_subset.kind") or "all")
    begin = OmegaConf.select(cfg, "trace_subset.begin")
    end = OmegaConf.select(cfg, "trace_subset.end")
    indices_val = OmegaConf.select(cfg, "trace_subset.indices")
    if indices_val is None:
        indices = None
    elif isinstance(indices_val, (list, tuple)):
        indices = list(indices_val)
    elif hasattr(indices_val, "__iter__") and not isinstance(indices_val, (str, bytes)):
        indices = list(indices_val)
    else:
        raise ValueError(f"trace_subset.indices must be a list (e.g. [0,3,10,42]) (got {indices_val!r})")
    return kind, begin, end, indices


def _run_trace(cfg: DictConfig, *, repo_root: Path) -> Path:
    try:
        preflight_trace(cfg, repo_root=repo_root)
    except MissingTraceSourceError as e:
        raise RuntimeError(
            f"{e}\n"
            "Hint: initialize submodules with `git submodule update --init --recursive` "
            "(the processed-lengths trace CSV lives under `extern/tracked/vidur`)."
        ) from e
    except ScenarioPreflightError as e:
        raise RuntimeError(str(e)) from e

    scenario_name = str(cfg.scenario.name)
    trace_kind = str(cfg.scenario.trace_source.kind)
    trace_source_path = Path(cfg.scenario.trace_source.path).expanduser()
    if not trace_source_path.is_absolute():
        trace_source_path = (repo_root / trace_source_path).resolve()
    else:
        trace_source_path = trace_source_path.resolve()

    max_tokens = int(cfg.scenario.trace_source.max_tokens)
    seed = int(cfg.scenario.trace_source.seed)
    num_requests = cfg.scenario.trace_source.num_requests
    spec = TraceSpec(
        max_tokens=max_tokens,
        seed=seed,
        num_requests=None if num_requests in (None, "null") else int(num_requests),
    )

    pf_paths = PaperFidelityPaths(repo_root=repo_root)
    out_dir = pf_paths.trace_dir(scenario_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_csv = out_dir / "trace.csv"

    subset_kind, subset_begin, subset_end, subset_indices = _trace_subset_from_cfg(cfg)

    if trace_kind == "vidur_processed_lengths_csv":
        df = processed_lengths_csv_to_trace(trace_source_path, spec=spec)
        df = apply_trace_subset(
            df,
            spec=spec,
            kind=subset_kind,
            begin=subset_begin,
            end=subset_end,
            indices=subset_indices,
            allow_indices=True,
            rebase_arrived_at=False,
        )
        if str(cfg.workload.mode) == "dynamic":
            df = add_poisson_arrivals(df, qps=float(cfg.workload.qps), seed=int(cfg.workload.seed))
        else:
            df = make_static(df)
    elif trace_kind == "trace_csv":
        df = read_trace_csv(trace_source_path, spec=spec)
        df = apply_trace_subset(
            df,
            spec=spec,
            kind=subset_kind,
            begin=subset_begin,
            end=subset_end,
            indices=subset_indices,
            allow_indices=False,
            rebase_arrived_at=True,
        )
        if str(cfg.workload.mode) == "static":
            df = make_static(df)
    elif trace_kind == "legacy_workload_dir":
        df = legacy_workload_dir_to_trace(trace_source_path, out_csv=trace_csv, spec=spec)
        df = apply_trace_subset(
            df,
            spec=spec,
            kind=subset_kind,
            begin=subset_begin,
            end=subset_end,
            indices=subset_indices,
            allow_indices=False,
            rebase_arrived_at=True,
        )
        if str(cfg.workload.mode) == "static":
            df = make_static(df)
    else:
        raise ValueError(f"Unsupported trace_source.kind: {trace_kind}")

    validate_trace(df, spec=spec)
    df.to_csv(trace_csv, index=False)

    trace_meta = {
        "schema_version": "v1",
        "scenario_name": scenario_name,
        "workload_mode": str(cfg.workload.mode),
        "scale": OmegaConf.select(cfg, "scale"),
        "trace_source": {
            "kind": trace_kind,
            "path": str(trace_source_path.resolve()),
            "max_tokens": max_tokens,
            "seed": seed,
            "num_requests": spec.num_requests,
        },
        "trace_subset": {
            "kind": subset_kind,
            "begin": subset_begin,
            "end": subset_end,
            "indices": subset_indices,
        },
        "generated_at": utcnow_iso(),
        "artifacts": {
            "trace_csv": str(trace_csv.resolve()),
        },
    }
    write_json(out_dir / "trace_meta.json", trace_meta)
    return out_dir


def _run_score_only(
    cfg: DictConfig,
    *,
    sim_csv: Path,
    real_csv: Path,
    repo_root: Path,
    profiling: dict[str, object] | None = None,
    profiling_meta_json: Path | None = None,
    report_scenario_name: str | None = None,
    trace_csv: Path | None = None,
    trace_meta_json: Path | None = None,
    capacity_json: Path | None = None,
) -> Path:
    sim_csv = sim_csv.expanduser()
    real_csv = real_csv.expanduser()

    if not sim_csv.is_absolute():
        sim_csv = (repo_root / sim_csv).resolve()
    else:
        sim_csv = sim_csv.resolve()

    if not real_csv.is_absolute():
        real_csv = (repo_root / real_csv).resolve()
    else:
        real_csv = real_csv.resolve()

    inferred_scenario_name = _infer_scenario_name(sim_csv=sim_csv, real_csv=real_csv)
    report_name = report_scenario_name or inferred_scenario_name
    if not str(report_name).strip():
        report_name = inferred_scenario_name
    pf_paths = PaperFidelityPaths(repo_root=repo_root)
    report_dir = pf_paths.reports_dir(date=_utc_date_str(), scenario_name=str(report_name))

    # Snapshot run inputs into the report directory so later runs do not overwrite the CSVs that
    # the report points at (tmp/ paths are intentionally reused for iteration).
    inputs_dir = report_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    sim_snapshot = inputs_dir / "sim_request_metrics.csv"
    real_snapshot = inputs_dir / "real_request_metrics.csv"
    shutil.copy2(sim_csv, sim_snapshot)
    shutil.copy2(real_csv, real_snapshot)

    trace_snapshot: Path | None = None
    if trace_csv is not None:
        trace_snapshot = inputs_dir / "trace.csv"
        shutil.copy2(trace_csv, trace_snapshot)

    trace_meta_snapshot: Path | None = None
    trace_meta_src = None
    if trace_meta_json is not None:
        trace_meta_src = trace_meta_json
    elif trace_csv is not None:
        candidate = trace_csv.parent / "trace_meta.json"
        if candidate.exists():
            trace_meta_src = candidate
    if trace_meta_src is not None and trace_meta_src.exists():
        trace_meta_snapshot = inputs_dir / "trace_meta.json"
        shutil.copy2(trace_meta_src, trace_meta_snapshot)

    capacity_snapshot: Path | None = None
    if capacity_json is not None and capacity_json.exists():
        capacity_snapshot = inputs_dir / "capacity.json"
        shutil.copy2(capacity_json, capacity_snapshot)

    profiling_meta_snapshot: Path | None = None
    if profiling_meta_json is not None and profiling_meta_json.exists():
        profiling_meta_snapshot = inputs_dir / "profiling_meta.json"
        shutil.copy2(profiling_meta_json, profiling_meta_snapshot)

    sim_df = load_metrics_csv(sim_snapshot)
    real_df = load_metrics_csv(real_snapshot)
    validate_sim_vs_real_compatibility(sim_df=sim_df, real_df=real_df)

    percentiles_val = OmegaConf.select(cfg, "scoring.percentiles") or OmegaConf.select(cfg, "scenario.scoring.percentiles")
    if percentiles_val is None:
        raise ValueError("Missing scoring.percentiles in config")
    percentiles = [float(q) for q in percentiles_val]

    pass_pct = OmegaConf.select(cfg, "scoring.thresholds.pass_pct") or OmegaConf.select(cfg, "scenario.scoring.thresholds.pass_pct")
    warn_pct = OmegaConf.select(cfg, "scoring.thresholds.warn_pct") or OmegaConf.select(cfg, "scenario.scoring.thresholds.warn_pct")
    if pass_pct is None or warn_pct is None:
        raise ValueError("Missing scoring.thresholds in config")

    thresholds = ScoreThresholds(
        pass_pct=float(pass_pct),
        warn_pct=float(warn_pct),
    )

    metrics = [
        # Request-level metrics (paper-fidelity main comparisons).
        "request_execution_plus_preemption_time_normalized",
        "request_e2e_time_normalized",
        # Stage-level metrics (prefill vs decode, token-normalized).
        "prefill_time_execution_plus_preemption_normalized",
        "decode_time_execution_plus_preemption_normalized",
    ]
    results = [
        score_metric(sim_df=sim_df, real_df=real_df, metric=m, percentiles=percentiles, thresholds=thresholds)
        for m in metrics
    ]

    git = get_git_info(repo_root=repo_root)
    meta: dict = {
        "schema_version": "v1",
        "run_type": "score",
        "run_id": stable_id([str(sim_csv), str(real_csv)], prefix="pf_score", length=12),
        "scenario_name": str(report_name),
        "base_scenario_name": inferred_scenario_name,
        "started_at": utcnow_iso(),
        "ended_at": utcnow_iso(),
        "git_commit": git.commit or "unknown",
        "git_dirty": git.dirty,
        "env": build_env_snapshot(),
        "params": OmegaConf.to_container(cfg, resolve=True),
        "artifacts": {
            "sim_csv_original": str(sim_csv),
            "real_csv_original": str(real_csv),
            "sim_csv": str(sim_snapshot),
            "real_csv": str(real_snapshot),
            "trace_csv": str(trace_snapshot) if trace_snapshot is not None else None,
            "trace_meta_json": str(trace_meta_snapshot) if trace_meta_snapshot is not None else None,
            "capacity_json": str(capacity_snapshot) if capacity_snapshot is not None else None,
            "profiling_meta_json": str(profiling_meta_snapshot) if profiling_meta_snapshot is not None else None,
            "report_dir": str(report_dir.resolve()),
        },
    }
    if profiling is not None:
        meta["profiling"] = profiling

    paper_reference_requested = bool(OmegaConf.select(cfg, "paper_reference.enabled") or False)
    if paper_reference_requested:
        workload_mode = OmegaConf.select(cfg, "workload.mode")
        load_frac = (
            OmegaConf.select(cfg, "scenario.paper_reference.dynamic.load_frac_of_capacity")
            if workload_mode == "dynamic"
            else None
        )
        paper_meta: dict[str, object] = {
            "schema_version": "v1",
            "requested": True,
            "matched": False,
            "workload_mode": str(workload_mode) if workload_mode is not None else None,
            "load_frac_of_capacity": float(load_frac) if load_frac is not None else None,
            "criteria": {
                "metric": OmegaConf.select(cfg, f"scenario.paper_reference.{workload_mode}.metric"),
                "model": OmegaConf.select(cfg, "scenario.paper_reference.model"),
                "trace": OmegaConf.select(cfg, "scenario.paper_reference.trace"),
                "series": OmegaConf.select(cfg, "scenario.paper_reference.series") or "predicted",
                "p50_json": OmegaConf.select(cfg, f"scenario.paper_reference.{workload_mode}.p50_json"),
                "p95_json": OmegaConf.select(cfg, f"scenario.paper_reference.{workload_mode}.p95_json"),
            },
            "rows": [],
            "error": None,
        }

        try:
            paper_rows = maybe_load_paper_reference_rows_from_cfg(cfg, repo_root=repo_root)
        except Exception as e:
            paper_meta["error"] = f"{type(e).__name__}: {e}"
        else:
            if paper_rows is None:
                paper_ref_cfg = OmegaConf.select(cfg, "scenario.paper_reference")
                if paper_ref_cfg is None:
                    paper_meta["error"] = "scenario.paper_reference is missing"
                elif not bool(OmegaConf.select(cfg, "scenario.paper_reference.enabled") or False):
                    paper_meta["error"] = "scenario.paper_reference.enabled=false"
                else:
                    paper_meta["error"] = "paper reference unavailable"
            else:
                paper_meta["matched"] = True
                paper_meta["rows"] = [
                    {
                        "metric": row.metric,
                        "percentile": row.percentile,
                        "value": row.value,
                        "model": row.model,
                        "trace": row.trace,
                        "series": row.series,
                        "source_json": str(row.source_json),
                        "source_pdf": row.source_pdf,
                    }
                    for row in paper_rows
                ]

        meta["paper_reference"] = paper_meta

    write_summary_md(
        inputs=ReportInputs(
            scenario_name=str(report_name),
            sim_csv=sim_snapshot,
            real_csv=real_snapshot,
            out_dir=report_dir,
        ),
        results=results,
        meta=meta,
    )
    return report_dir


def _run_repro(cfg: DictConfig, *, repo_root: Path) -> Path:
    from gpu_simulate_test.env_guard import (
        apply_cuda_visible_devices_from_gsim,
        count_visible_gpus,
        patch_sarathi_preserve_cuda_visible_devices,
    )

    apply_cuda_visible_devices_from_gsim(repo_root=repo_root)
    patch_sarathi_preserve_cuda_visible_devices()
    preflight_repro(cfg, repo_root=repo_root, available_gpus=count_visible_gpus())

    scenario_name = str(cfg.scenario.name)
    workload_mode = str(cfg.workload.mode)
    scale = str(OmegaConf.select(cfg, "scale") or "full")
    pf_paths = PaperFidelityPaths(repo_root=repo_root)

    ignore_eos_val = OmegaConf.select(cfg, "scenario.real.sampling.ignore_eos")
    ignore_eos = bool(ignore_eos_val) if ignore_eos_val is not None else True

    started_at = utcnow_iso()
    profiling_root_raw = Path(cfg.scenario.vidur.profiling_root).expanduser()
    if profiling_root_raw.is_absolute():
        profiling_root_resolved = profiling_root_raw.resolve()
    else:
        profiling_root_resolved = (repo_root / profiling_root_raw).resolve()

    paper_profiling_root = (repo_root / "extern" / "tracked" / "vidur").resolve()
    host_profiling_root = (repo_root / "tmp" / "paper_fidelity" / "profiling_roots").resolve()
    host_bundle_root = (repo_root / "results" / "raw" / "vidur-profiling").resolve()
    if profiling_root_resolved.is_relative_to(paper_profiling_root):
        profiling_mode = "paper"
        profiling_interpretation = "sanity-check reproduction (paper-provided profiling bundle)"
    elif profiling_root_resolved.is_relative_to(host_profiling_root):
        profiling_mode = "host"
        profiling_interpretation = "gap reproduction (profiled/microbenchmarked on this host)"
    elif profiling_root_resolved.is_relative_to(host_bundle_root):
        profiling_mode = "host"
        profiling_interpretation = "gap reproduction (host profiling bundle under results/raw/vidur-profiling)"
    else:
        profiling_mode = "custom"
        profiling_interpretation = "custom profiling root (interpret % error accordingly)"

    report_name = scenario_name if workload_mode == "static" else f"{scenario_name}_{workload_mode}_{scale}"
    capacity_json: Path | None = None

    if workload_mode == "dynamic":
        capacity_cfg = cfg.scenario.capacity_search
        capacity_dir = pf_paths.capacity_dir(scenario_name)
        capacity_dir.mkdir(parents=True, exist_ok=True)

        num_requests = cfg.scenario.trace_source.num_requests
        trace_spec = TraceSpec(
            max_tokens=int(cfg.scenario.trace_source.max_tokens),
            seed=int(cfg.scenario.trace_source.seed),
            num_requests=None if num_requests in (None, "null") else int(num_requests),
        )

        base = None
        trace_kind = str(cfg.scenario.trace_source.kind)
        trace_source_path = Path(cfg.scenario.trace_source.path).expanduser()
        if not trace_source_path.is_absolute():
            trace_source_path = (repo_root / trace_source_path).resolve()
        else:
            trace_source_path = trace_source_path.resolve()
        if trace_kind == "vidur_processed_lengths_csv":
            base = processed_lengths_csv_to_trace(trace_source_path, spec=trace_spec)
        else:
            raise ValueError(f"capacity discovery requires vidur_processed_lengths_csv (got {trace_kind})")

        subset_kind, subset_begin, subset_end, subset_indices = _trace_subset_from_cfg(cfg)
        base = apply_trace_subset(
            base,
            spec=trace_spec,
            kind=subset_kind,
            begin=subset_begin,
            end=subset_end,
            indices=subset_indices,
            allow_indices=True,
            rebase_arrived_at=False,
        )

        criterion = CapacityCriterion(
            metric="request_scheduling_delay",
            quantile=0.99,
            threshold_s=float(capacity_cfg.overload_p99_scheduling_delay_s),
        )

        def run_at_qps(qps: float) -> pd.DataFrame:
            trace_df = add_poisson_arrivals(base, qps=float(qps), seed=int(cfg.workload.seed))
            trace_csv = capacity_dir / "trace.csv"
            trace_df.to_csv(trace_csv, index=False)
            out_dir = capacity_dir / f"qps_{qps:.4f}"
            req_csv = run_sarathi_paper_fidelity(
                SarathiPaperFidelityInputs(
                    scenario_name=scenario_name,
                    trace_csv=trace_csv,
                    model_id=str(cfg.scenario.model.model_id),
                    model_ref=Path(cfg.scenario.model.model_ref).expanduser(),
                    seed=int(cfg.scenario.real.get("seed", 42)) if "real" in cfg.scenario else 42,
                    max_tokens=int(cfg.scenario.trace_source.max_tokens),
                    chunk_size=int(cfg.scenario.real.scheduler.chunk_size),
                    max_num_seqs=int(cfg.scenario.real.scheduler.max_num_seqs),
                    tensor_parallel_size=int(cfg.scenario.real.parallel.tensor_parallel_size),
                    pipeline_parallel_size=int(cfg.scenario.real.parallel.pipeline_parallel_size),
                    ignore_eos=ignore_eos,
                ),
                out_dir=out_dir,
            )
            return pd.read_csv(req_csv)

        capacity = discover_capacity(
            run_at_qps=run_at_qps,
            min_qps=float(capacity_cfg.min_qps),
            max_qps=float(capacity_cfg.max_qps),
            max_iters=int(capacity_cfg.max_iters),
            criterion=criterion,
            operating_point_fraction=float(capacity_cfg.qps_operating_point_fraction),
        )
        write_capacity_json(capacity_dir / "capacity.json", result=capacity)
        capacity_json = capacity_dir / "capacity.json"
        write_json(
            capacity_dir / "run_meta.json",
            {
                "schema_version": "v1",
                "run_type": "capacity",
                "run_id": stable_id([scenario_name, workload_mode], prefix="pf_capacity", length=12),
                "started_at": started_at,
                "ended_at": utcnow_iso(),
                "capacity": capacity.to_dict(),
            },
        )

        trace_dir = pf_paths.trace_dir(scenario_name)
        trace_dir.mkdir(parents=True, exist_ok=True)
        final_trace = add_poisson_arrivals(base, qps=float(capacity.qps_85), seed=int(cfg.workload.seed))
        validate_trace(final_trace, spec=trace_spec)
        trace_csv = trace_dir / "trace.csv"
        final_trace.to_csv(trace_csv, index=False)
        max_tokens = int(cfg.scenario.trace_source.max_tokens)
        seed = int(cfg.scenario.trace_source.seed)
        write_json(
            trace_dir / "trace_meta.json",
            {
                "schema_version": "v1",
                "scenario_name": scenario_name,
                "workload_mode": workload_mode,
                "scale": OmegaConf.select(cfg, "scale"),
                "qps": float(capacity.qps_85),
                "seed": int(cfg.workload.seed),
                "trace_source": {
                    "kind": trace_kind,
                    "path": str(trace_source_path),
                    "max_tokens": max_tokens,
                    "seed": seed,
                    "num_requests": trace_spec.num_requests,
                },
                "trace_subset": {
                    "kind": subset_kind,
                    "begin": subset_begin,
                    "end": subset_end,
                    "indices": subset_indices,
                },
                "generated_at": utcnow_iso(),
                "artifacts": {
                    "trace_csv": str(trace_csv.resolve()),
                },
            },
        )
    else:
        trace_dir = _run_trace(cfg, repo_root=repo_root)
        trace_csv = trace_dir / "trace.csv"
    trace_meta_json = trace_dir / "trace_meta.json"

    git = get_git_info(repo_root=repo_root)

    sim_dir = pf_paths.sim_dir(scenario_name)
    sim_run_meta = {
        "schema_version": "v1",
        "run_type": "sim",
        "run_id": stable_id([scenario_name, workload_mode], prefix="pf_sim", length=12),
        "scenario_name": scenario_name,
        "started_at": started_at,
        "git_commit": git.commit or "unknown",
        "git_dirty": git.dirty,
        "env": build_env_snapshot(),
        "params": OmegaConf.to_container(cfg, resolve=True),
    }
    skip_val = OmegaConf.select(cfg, "scenario.vidur.skip_cpu_overhead_modeling")  # back-compat
    enable_val = OmegaConf.select(cfg, "scenario.vidur.enable_cpu_overhead_modeling")  # back-compat alias
    if skip_val is not None:
        skip_cpu_overhead_modeling = bool(skip_val)
    elif enable_val is not None:
        skip_cpu_overhead_modeling = not bool(enable_val)
    else:
        skip_cpu_overhead_modeling = True

    cpu_validation_val = OmegaConf.select(cfg, "scenario.vidur.cpu_overhead.validation")
    if cpu_validation_val is None:
        cpu_validation_val = OmegaConf.select(cfg, "scenario.vidur.cpu_overhead_validation")  # back-compat
    cpu_overhead_validation = (
        str(cpu_validation_val).lower().strip() if cpu_validation_val is not None else "strict"
    )

    scheduler_type = str(OmegaConf.select(cfg, "scenario.vidur.scheduler.type") or "sarathi")
    scheduler_chunk_size = OmegaConf.select(cfg, "scenario.vidur.scheduler.chunk_size")
    scheduler_batch_size_cap = OmegaConf.select(cfg, "scenario.vidur.scheduler.batch_size_cap")
    scheduler_block_size = OmegaConf.select(cfg, "scenario.vidur.scheduler.block_size")
    scheduler_watermark = OmegaConf.select(cfg, "scenario.vidur.scheduler.watermark_blocks_fraction")
    run_vidur_paper_fidelity_sim(
        VidurPaperFidelitySimInputs(
            scenario_name=scenario_name,
            trace_csv=trace_csv,
            profiling_root=profiling_root_resolved,
            model_id=str(cfg.scenario.model.model_id),
            device=str(cfg.scenario.vidur.device),
            network_device=str(cfg.scenario.vidur.network_device),
            tensor_parallel_size=int(cfg.scenario.vidur.tensor_parallel_size),
            num_pipeline_stages=int(cfg.scenario.vidur.num_pipeline_stages),
            seed=int(cfg.scenario.vidur.seed),
            max_tokens=int(cfg.scenario.trace_source.max_tokens),
            skip_cpu_overhead_modeling=skip_cpu_overhead_modeling,
            cpu_overhead_validation=cpu_overhead_validation,
            scheduler_type=scheduler_type,
            scheduler_chunk_size=None if scheduler_chunk_size is None else int(scheduler_chunk_size),
            scheduler_batch_size_cap=None if scheduler_batch_size_cap is None else int(scheduler_batch_size_cap),
            scheduler_block_size=None if scheduler_block_size is None else int(scheduler_block_size),
            scheduler_watermark_blocks_fraction=None
            if scheduler_watermark is None
            else float(scheduler_watermark),
        ),
        out_dir=sim_dir,
        run_meta=sim_run_meta,
    )

    real_dir = pf_paths.real_dir(scenario_name)
    run_sarathi_paper_fidelity(
        SarathiPaperFidelityInputs(
            scenario_name=scenario_name,
            trace_csv=trace_csv,
            model_id=str(cfg.scenario.model.model_id),
            model_ref=Path(cfg.scenario.model.model_ref).expanduser(),
            seed=int(cfg.scenario.real.get("seed", 42)) if "real" in cfg.scenario else 42,
            max_tokens=int(cfg.scenario.trace_source.max_tokens),
            chunk_size=int(cfg.scenario.real.scheduler.chunk_size),
            max_num_seqs=int(cfg.scenario.real.scheduler.max_num_seqs),
            tensor_parallel_size=int(cfg.scenario.real.parallel.tensor_parallel_size),
            pipeline_parallel_size=int(cfg.scenario.real.parallel.pipeline_parallel_size),
            ignore_eos=ignore_eos,
        ),
        out_dir=real_dir,
    )

    report_dir = _run_score_only(
        cfg,
        sim_csv=sim_dir / "request_metrics.csv",
        real_csv=real_dir / "request_metrics.csv",
        repo_root=repo_root,
        report_scenario_name=report_name,
        trace_csv=trace_csv,
        trace_meta_json=trace_meta_json,
        capacity_json=capacity_json,
        profiling_meta_json=profiling_root_resolved / "profiling_meta.json",
        profiling={
            "root": str(profiling_root_resolved),
            "mode": profiling_mode,
            "interpretation": profiling_interpretation,
            "cpu_overhead": _inspect_cpu_overhead_inputs(
                profiling_root=profiling_root_resolved,
                model_id=str(cfg.scenario.model.model_id),
                network_device=str(cfg.scenario.vidur.network_device),
                skip_cpu_overhead_modeling=bool(skip_cpu_overhead_modeling),
                validation_mode=cpu_overhead_validation,
                expected_tensor_parallel_size=int(cfg.scenario.vidur.tensor_parallel_size),
            ),
        },
    )
    return report_dir


def _run_profile(cfg: DictConfig, *, repo_root: Path) -> Path:
    from gpu_simulate_test.env_guard import (
        apply_cuda_visible_devices_from_gsim,
        patch_sarathi_preserve_cuda_visible_devices,
    )

    apply_cuda_visible_devices_from_gsim(repo_root=repo_root)
    patch_sarathi_preserve_cuda_visible_devices()

    return run_paper_fidelity_profiling(cfg, repo_root=repo_root)


@hydra.main(
    config_path="../../../configs/paper_fidelity",
    config_name="repro",
    version_base=None,
)
def _repro_main(cfg: DictConfig) -> None:
    """Hydra main for `paper-fidelity repro` (see `configs/paper_fidelity/repro.yaml`)."""
    repo_root = Path(cfg.paths.repo_root)
    scenario_name = str(cfg.scenario.name)
    workload_mode = str(cfg.workload.mode)
    scale = str(OmegaConf.select(cfg, "scale") or "full")
    report_name = scenario_name if workload_mode == "static" else f"{scenario_name}_{workload_mode}_{scale}"

    pf_paths = PaperFidelityPaths(repo_root=repo_root)
    report_dir = pf_paths.reports_dir(date=_utc_date_str(), scenario_name=report_name)
    report_dir.mkdir(parents=True, exist_ok=True)

    scenario_key = str(OmegaConf.select(cfg, "hydra.runtime.choices.scenario") or scenario_name)

    try:
        out_dir = _run_repro(cfg, repo_root=repo_root)
    except Exception as e:
        import traceback as tb

        stack = tb.format_exc()
        category = categorize_blocker(error_message=f"{type(e).__name__}: {e}", traceback=stack)
        record = build_failure_record(
            run_id=stable_id([scenario_key, report_name, workload_mode, scale], prefix="pf_repro", length=12),
            action="repro",
            scenario_key=scenario_key,
            scenario_name=report_name,
            workload=workload_mode,
            scale=scale,
            attempted_command=list(sys.argv),
            hydra_overrides=[],
            error_message=f"{type(e).__name__}: {e}",
            traceback=stack,
            blocker_category=category,
        )
        failure_path = write_failure_record(report_dir / "failure_record.json", record)
        print(str(failure_path))
        raise PaperFidelityReproError(f"{type(e).__name__}: {e}", failure_record_path=failure_path) from e
    else:
        print(str(out_dir))


@hydra.main(
    config_path="../../../configs/paper_fidelity",
    config_name="trace",
    version_base=None,
)
def _trace_main(cfg: DictConfig) -> None:
    """Hydra main for `paper-fidelity trace` (see `configs/paper_fidelity/trace.yaml`)."""
    out_dir = _run_trace(cfg, repo_root=Path(cfg.paths.repo_root))
    print(str(out_dir))


@hydra.main(
    config_path="../../../configs/paper_fidelity",
    config_name="profile",
    version_base=None,
)
def _profile_main(cfg: DictConfig) -> None:
    """Hydra main for `paper-fidelity profile` (see `configs/paper_fidelity/profile.yaml`)."""
    try:
        out_dir = _run_profile(cfg, repo_root=Path(cfg.paths.repo_root))
    except PaperFidelityProfilingError as e:
        print(str(e.failure_record_path))
        raise
    else:
        print(str(out_dir))


@hydra.main(
    config_path="../../../configs/paper_fidelity",
    config_name="score",
    version_base=None,
)
def _score_main(cfg: DictConfig) -> None:
    """Hydra main for `paper-fidelity score` (see `configs/paper_fidelity/score.yaml`)."""
    sim_csv = Path(cfg.inputs.sim_csv)
    real_csv = Path(cfg.inputs.real_csv)
    out_dir = _run_score_only(cfg, sim_csv=sim_csv, real_csv=real_csv, repo_root=Path(cfg.paths.repo_root))
    print(str(out_dir))


if __name__ == "__main__":
    main()
