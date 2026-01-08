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
from gpu_simulate_test.paper_fidelity.paper_reference import maybe_load_paper_reference_rows_from_cfg
from gpu_simulate_test.paper_fidelity.paths import PaperFidelityPaths
from gpu_simulate_test.paper_fidelity.profiling import run_paper_fidelity_profiling
from gpu_simulate_test.paper_fidelity.report import ReportInputs, regenerate_summary_md_from_report_dir, write_summary_md
from gpu_simulate_test.paper_fidelity.scoring import (
    ScoreThresholds,
    load_metrics_csv,
    score_metric,
    validate_sim_vs_real_compatibility,
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
    scenario_name = str(cfg.scenario.name)
    trace_kind = str(cfg.scenario.trace_source.kind)
    trace_source_path = Path(cfg.scenario.trace_source.path).expanduser()

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

    scenario_name = _infer_scenario_name(sim_csv=sim_csv, real_csv=real_csv)
    pf_paths = PaperFidelityPaths(repo_root=repo_root)
    report_dir = pf_paths.reports_dir(date=_utc_date_str(), scenario_name=scenario_name)

    sim_df = load_metrics_csv(sim_csv)
    real_df = load_metrics_csv(real_csv)
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
        "scenario_name": scenario_name,
        "started_at": utcnow_iso(),
        "ended_at": utcnow_iso(),
        "git_commit": git.commit or "unknown",
        "git_dirty": git.dirty,
        "env": build_env_snapshot(),
        "params": OmegaConf.to_container(cfg, resolve=True),
        "artifacts": {
            "sim_csv": str(sim_csv),
            "real_csv": str(real_csv),
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
        inputs=ReportInputs(scenario_name=scenario_name, sim_csv=sim_csv, real_csv=real_csv, out_dir=report_dir),
        results=results,
        meta=meta,
    )
    return report_dir


def _run_repro(cfg: DictConfig, *, repo_root: Path) -> Path:
    scenario_name = str(cfg.scenario.name)
    workload_mode = str(cfg.workload.mode)
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
        write_json(
            trace_dir / "trace_meta.json",
            {
                "schema_version": "v1",
                "scenario_name": scenario_name,
                "workload_mode": workload_mode,
                "scale": OmegaConf.select(cfg, "scale"),
                "qps": float(capacity.qps_85),
                "seed": int(cfg.workload.seed),
                "trace_subset": {
                    "kind": subset_kind,
                    "begin": subset_begin,
                    "end": subset_end,
                    "indices": subset_indices,
                },
                "generated_at": utcnow_iso(),
            },
        )
    else:
        trace_dir = _run_trace(cfg, repo_root=repo_root)
        trace_csv = trace_dir / "trace.csv"

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
    skip_val = OmegaConf.select(cfg, "scenario.vidur.skip_cpu_overhead_modeling")
    skip_cpu_overhead_modeling = bool(skip_val) if skip_val is not None else True
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
        profiling={
            "root": str(profiling_root_resolved),
            "mode": profiling_mode,
            "interpretation": profiling_interpretation,
        },
    )
    return report_dir


def _run_profile(cfg: DictConfig, *, repo_root: Path) -> Path:
    return run_paper_fidelity_profiling(cfg, repo_root=repo_root)


@hydra.main(
    config_path="../../../configs/paper_fidelity",
    config_name="repro",
    version_base=None,
)
def _repro_main(cfg: DictConfig) -> None:
    """Hydra main for `paper-fidelity repro` (see `configs/paper_fidelity/repro.yaml`)."""
    out_dir = _run_repro(cfg, repo_root=Path(cfg.paths.repo_root))
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
    out_dir = _run_profile(cfg, repo_root=Path(cfg.paths.repo_root))
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
