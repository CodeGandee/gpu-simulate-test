"""
Stage runners for `vidur-cli`.

This module contains the implementation of `svr` subcommands as independent,
resumable steps operating on a run directory.

Functions
---------
run_init_run
    Create a run directory and initialize `run_state.json` + `resources.json`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from gpu_simulate_test.io import utcnow_iso
from gpu_simulate_test.vidur_cli.errors import UserFacingError
from gpu_simulate_test.vidur_cli.real_runner import run_token_length_replay
from gpu_simulate_test.vidur_cli.resources import ResourceMapV1, write_resources_json
from gpu_simulate_test.vidur_cli.run_state import (
    Presets,
    load_run_state,
    require_profile_artifacts,
    require_real_artifacts,
    require_sim_artifacts,
    require_trace_artifacts,
    run_with_failure_json,
    save_run_state,
    write_run_state,
)
from gpu_simulate_test.vidur_cli.search_path import compose_config
from gpu_simulate_test.vidur_cli.trace import TraceBuildResult, build_default_trace, build_from_lengths_csv, import_canonical_trace
from gpu_simulate_test.vidur_ext.profile_runner import VidurProfileInputs, run_vidur_profiling
from gpu_simulate_test.vidur_ext.qwen3_model_config import Qwen3ModelRef, register_qwen3_0_6b
from gpu_simulate_test.vidur_ext.sim_runner import VidurSimInputs, run_vidur_sim
from gpu_simulate_test.workloads.arrival_schedule import ArrivalScheduleConfig


def run_init_run(
    *,
    run_dir: Path,
    run_tag: str,
    presets: Presets,
    overrides: list[str],
    resources: ResourceMapV1,
) -> Path:
    """Create the run directory and initialize v1 run artifacts."""
    run_dir.mkdir(parents=True, exist_ok=True)

    def _impl() -> Path:
        write_resources_json(run_dir=run_dir, resources=resources)
        write_run_state(run_dir=run_dir, presets=presets, overrides=overrides, run_tag=run_tag)
        _write_resolved_config_yaml(
            run_dir=run_dir,
            run_tag=run_tag,
            presets=presets,
            overrides=overrides,
            resources=resources,
        )
        return run_dir.resolve()

    return run_with_failure_json(run_dir=run_dir, stage="init-run", fn=_impl, failure_context={"run_tag": run_tag})


def run_trace(
    *,
    run_dir: Path,
    resources: ResourceMapV1,
    overrides: list[str],
    import_trace: Path | None,
    from_lengths: Path | None,
) -> Path:
    """Materialize canonical trace artifacts under `<run_dir>/trace/`."""
    run_dir = run_dir.expanduser().resolve()

    trace_dir = run_dir / "trace"
    trace_csv = trace_dir / "trace.csv"
    trace_meta = trace_dir / "trace_meta.json"

    def _on_error(run_state: dict[str, Any], ended_at: str) -> dict[str, Any]:
        artifacts = dict(run_state.get("artifacts") or {})
        artifacts["trace"] = {
            "trace_csv": str(trace_csv.resolve()),
            "trace_meta_json": str(trace_meta.resolve()),
            "status": "failed",
            "ended_at": ended_at,
            "overrides": list(overrides),
        }
        run_state["artifacts"] = artifacts
        return run_state

    def _impl() -> Path:
        state = load_run_state(run_dir=run_dir)
        schedule, prompts_jsonl, tokenizer_ref, num_decode_tokens = _load_workload_inputs(
            state=state, resources=resources, overrides=overrides
        )
        trace_dir.mkdir(parents=True, exist_ok=True)

        if import_trace is not None and from_lengths is not None:
            raise UserFacingError("Only one of --import-trace or --from-lengths may be provided.")

        if import_trace is not None:
            import_path = import_trace.expanduser()
            if not import_path.exists():
                raise UserFacingError(
                    f"--import-trace does not exist: {import_path}",
                    hint="Provide a valid CSV path (canonical trace schema).",
                )
            result = import_canonical_trace(src_csv=import_trace, out_dir=trace_dir)
        elif from_lengths is not None:
            lengths_path = from_lengths.expanduser()
            if not lengths_path.exists():
                raise UserFacingError(
                    f"--from-lengths does not exist: {lengths_path}",
                    hint="Provide a valid lengths-only CSV with num_prefill_tokens,num_decode_tokens columns.",
                )
            result = build_from_lengths_csv(lengths_csv=from_lengths, schedule=schedule, out_dir=trace_dir)
        else:
            prompts_path = _maybe_materialize_default_prompts(prompts_jsonl, resources=resources)
            result = build_default_trace(
                prompts_jsonl=prompts_path,
                tokenizer_ref=tokenizer_ref,
                num_decode_tokens=num_decode_tokens,
                schedule=schedule,
                out_dir=trace_dir,
            )

        _update_run_state_trace_ok(run_dir=run_dir, result=result, overrides=overrides)
        return result.trace_csv

    return run_with_failure_json(
        run_dir=run_dir,
        stage="trace",
        fn=_impl,
        failure_context={"run_dir": str(run_dir)},
        on_error=_on_error,
    )


def run_profile(
    *,
    run_dir: Path,
    resources: ResourceMapV1,
    overrides: list[str],
    include_cpu_overhead: bool,
) -> Path:
    """Run Vidur profiling and record the profiling root in `run_state.json`."""
    run_dir = run_dir.expanduser().resolve()
    out_dir = run_dir / "profile"

    def _on_error(run_state: dict[str, Any], ended_at: str) -> dict[str, Any]:
        artifacts = dict(run_state.get("artifacts") or {})
        artifacts["profile"] = {
            "profiling_root": str(out_dir.resolve()),
            "include_cpu_overhead": bool(include_cpu_overhead),
            "status": "failed",
            "ended_at": ended_at,
            "overrides": list(overrides),
        }
        run_state["artifacts"] = artifacts
        return run_state

    def _impl() -> Path:
        state = load_run_state(run_dir=run_dir)
        cfg = _compose_stage_cfg(
            config_name="vidur_profile",
            allowed_groups={"model", "hardware", "vidur"},
            state=state,
            resources=resources,
            overrides=overrides,
        )
        model_id = str(cfg.model.model_id)
        model_ref = Path(str(cfg.model.tokenizer_ref)).expanduser()
        hardware_id = str(cfg.hardware.hardware_id)
        _maybe_register_vidur_model(model_key=str((state.get("presets") or {}).get("model")), cfg=cfg)

        mlp_profile_method_val = OmegaConf.select(cfg, "profiling.mlp.profile_method")
        if mlp_profile_method_val is None:
            raise UserFacingError(
                "profiling.mlp.profile_method is required (no default).",
                hint="Pass an explicit override, e.g. profiling.mlp.profile_method=cuda_event",
            )
        mlp_profile_method = str(mlp_profile_method_val).strip()

        mlp_validation_mode_val = OmegaConf.select(cfg, "profiling.mlp.validation.mode")
        mlp_validation_mode = str(mlp_validation_mode_val or "strict").lower().strip()
        if mlp_validation_mode not in {"strict", "non_strict"}:
            raise UserFacingError(
                f"profiling.mlp.validation.mode must be 'strict' or 'non_strict' (got {mlp_validation_mode!r})."
            )

        mlp_nan_policy_val = OmegaConf.select(cfg, "profiling.mlp.validation.nan_policy")
        mlp_nan_policy = str(mlp_nan_policy_val or "auto").lower().strip()
        if mlp_nan_policy not in {"auto", "reject", "drop"}:
            raise UserFacingError(
                "profiling.mlp.validation.nan_policy must be one of auto|reject|drop "
                f"(got {mlp_nan_policy!r})."
            )

        mlp_small_input_threshold_val = OmegaConf.select(
            cfg, "profiling.mlp.validation.small_input_threshold"
        )
        mlp_small_input_threshold = int(mlp_small_input_threshold_val or 128)

        mlp_zero_heavy_limit_val = OmegaConf.select(cfg, "profiling.mlp.validation.zero_heavy_limit")
        mlp_zero_heavy_limit = float(mlp_zero_heavy_limit_val or 0.01)

        mlp_fallback_enabled_val = OmegaConf.select(cfg, "profiling.mlp.fallback.enabled")
        mlp_fallback_enabled = bool(mlp_fallback_enabled_val or False)

        mlp_fallback_method_val = OmegaConf.select(cfg, "profiling.mlp.fallback.method")
        mlp_fallback_method = str(mlp_fallback_method_val or "cuda_event").strip()

        out_dir.mkdir(parents=True, exist_ok=True)
        inputs = VidurProfileInputs(
            model_id=model_id,
            hardware_id=hardware_id,
            profiling_root=out_dir,
            mlp_profile_method=mlp_profile_method,
            mlp_validation_mode=mlp_validation_mode,  # type: ignore[arg-type]
            mlp_nan_policy=mlp_nan_policy,  # type: ignore[arg-type]
            mlp_small_input_threshold=mlp_small_input_threshold,
            mlp_zero_heavy_limit=mlp_zero_heavy_limit,
            mlp_fallback_enabled=mlp_fallback_enabled,
            mlp_fallback_method=mlp_fallback_method,
            include_cpu_overhead=bool(include_cpu_overhead),
            model_ref=model_ref,
        )
        result = run_vidur_profiling(inputs, repo_root=resources.repo_root.value)

        mlp_summary: dict[str, Any] = {
            "requested_profile_method": str(mlp_profile_method),
            "profile_method": str(result.extra.get("mlp_profile_method", mlp_profile_method)),
            "validation_mode": str(mlp_validation_mode),
            "nan_policy": str(mlp_nan_policy),
            "small_input_threshold": int(mlp_small_input_threshold),
            "zero_heavy_limit": float(mlp_zero_heavy_limit),
            "fallback_enabled": bool(mlp_fallback_enabled),
            "fallback_method": str(mlp_fallback_method),
            "fallback_used": bool(
                (result.extra.get("mlp_fallback") or {}).get("used", False)
                if isinstance(result.extra.get("mlp_fallback"), dict)
                else False
            ),
        }
        if "mlp_validation" in result.extra:
            mlp_summary["validation"] = result.extra["mlp_validation"]

        _update_run_state_profile_ok(
            run_dir=run_dir,
            profiling_root=out_dir,
            include_cpu_overhead=bool(include_cpu_overhead),
            overrides=overrides,
            mlp=mlp_summary,
        )
        return out_dir.resolve()

    return run_with_failure_json(
        run_dir=run_dir,
        stage="profile",
        fn=_impl,
        failure_context={"run_dir": str(run_dir), "out_dir": str(out_dir)},
        on_error=_on_error,
    )


def run_sim(
    *,
    run_dir: Path,
    resources: ResourceMapV1,
    overrides: list[str],
) -> Path:
    """Run Vidur simulation and record the sim output directory in `run_state.json`."""
    run_dir = run_dir.expanduser().resolve()
    out_dir = run_dir / "sim"

    def _on_error(run_state: dict[str, Any], ended_at: str) -> dict[str, Any]:
        artifacts = dict(run_state.get("artifacts") or {})
        artifacts["sim"] = {
            "sim_run_dir": str(out_dir.resolve()),
            "status": "failed",
            "ended_at": ended_at,
            "overrides": list(overrides),
        }
        run_state["artifacts"] = artifacts
        return run_state

    def _impl() -> Path:
        state = load_run_state(run_dir=run_dir)
        _ = require_trace_artifacts(run_dir=run_dir)
        profile = require_profile_artifacts(run_dir=run_dir)

        cfg = _compose_stage_cfg(
            config_name="vidur_sim",
            allowed_groups={"model", "hardware", "backend", "workload", "vidur"},
            state=state,
            resources=resources,
            overrides=overrides,
        )

        model_id = str(cfg.model.model_id)
        hardware_id = str(cfg.hardware.hardware_id)
        backend_name = str(cfg.backend.name)
        tokenizer_ref = Path(str(cfg.model.tokenizer_ref)).expanduser()
        _maybe_register_vidur_model(model_key=str((state.get("presets") or {}).get("model")), cfg=cfg)

        scheduler_chunk_size: int | None = None
        scheduler_batch_size_cap: int | None = None
        scheduler_block_size: int | None = None
        scheduler_watermark_blocks_fraction: float | None = None
        max_tokens: int = 4096
        if backend_name == "sarathi":
            scheduler = getattr(cfg.backend, "scheduler", None)
            if scheduler is not None:
                scheduler_chunk_size = int(getattr(scheduler, "chunk_size"))
                scheduler_batch_size_cap = int(getattr(scheduler, "max_num_seqs"))
                scheduler_block_size = int(getattr(scheduler, "block_size"))
                scheduler_watermark_blocks_fraction = float(getattr(scheduler, "watermark_blocks_fraction"))

            max_tokens = int(getattr(cfg.backend, "max_tokens"))

        mlp_validation_mode_val = OmegaConf.select(cfg, "vidur.validation.mlp.mode")
        mlp_validation_mode = str(mlp_validation_mode_val or "strict").lower().strip()
        if mlp_validation_mode not in {"strict", "non_strict"}:
            raise UserFacingError(
                f"vidur.validation.mlp.mode must be 'strict' or 'non_strict' (got {mlp_validation_mode!r})."
            )

        mlp_nan_policy_val = OmegaConf.select(cfg, "vidur.validation.mlp.nan_policy")
        mlp_nan_policy = str(mlp_nan_policy_val or "auto").lower().strip()
        if mlp_nan_policy not in {"auto", "reject", "drop"}:
            raise UserFacingError(
                "vidur.validation.mlp.nan_policy must be one of auto|reject|drop "
                f"(got {mlp_nan_policy!r})."
            )

        mlp_small_input_threshold_val = OmegaConf.select(cfg, "vidur.validation.mlp.small_input_threshold")
        mlp_small_input_threshold = int(mlp_small_input_threshold_val or 128)

        mlp_zero_heavy_limit_val = OmegaConf.select(cfg, "vidur.validation.mlp.zero_heavy_limit")
        mlp_zero_heavy_limit = float(mlp_zero_heavy_limit_val or 0.01)

        out_dir.mkdir(parents=True, exist_ok=True)
        inputs = VidurSimInputs(
            workload_dir=run_dir / "trace",
            profiling_root=profile.profiling_root,
            model_id=model_id,
            device=hardware_id,
            mlp_validation_mode=mlp_validation_mode,
            mlp_nan_policy=mlp_nan_policy,
            mlp_small_input_threshold=mlp_small_input_threshold,
            mlp_zero_heavy_limit=mlp_zero_heavy_limit,
            max_tokens=int(max_tokens),
            skip_cpu_overhead_modeling=not bool(profile.include_cpu_overhead),
            scheduler_chunk_size=scheduler_chunk_size,
            scheduler_batch_size_cap=scheduler_batch_size_cap,
            scheduler_block_size=scheduler_block_size,
            scheduler_watermark_blocks_fraction=scheduler_watermark_blocks_fraction,
        )
        run_meta = {
            "schema_version": "v1",
            "run_type": "vidur",
            "started_at": utcnow_iso(),
            "model_id": str(model_id),
            "model_ref": str(tokenizer_ref.resolve()),
            "hardware_id": str(hardware_id),
            "network_device": "a100_pairwise_nvlink",
            "max_tokens": int(max_tokens),
            "tensor_parallel_size": 1,
            "num_pipeline_stages": 1,
            "profiling_root": str(profile.profiling_root),
            "cpu_overhead": {
                "include_cpu_overhead": bool(profile.include_cpu_overhead),
                "skip_cpu_overhead_modeling": not bool(profile.include_cpu_overhead),
            },
            "scheduler": {
                "type": "sarathi",
                "chunk_size": scheduler_chunk_size,
                "batch_size_cap": scheduler_batch_size_cap,
                "block_size": scheduler_block_size,
                "watermark_blocks_fraction": scheduler_watermark_blocks_fraction,
            },
        }
        run_vidur_sim(inputs, out_dir=out_dir, run_meta=run_meta)

        paper_fidelity_csv = _maybe_write_sim_paper_fidelity_metrics(sim_run_dir=out_dir)
        _update_run_state_sim_ok(
            run_dir=run_dir,
            sim_run_dir=out_dir,
            paper_fidelity_request_metrics_csv=paper_fidelity_csv,
            overrides=overrides,
        )
        return out_dir.resolve()

    return run_with_failure_json(
        run_dir=run_dir,
        stage="sim",
        fn=_impl,
        failure_context={"run_dir": str(run_dir), "out_dir": str(out_dir)},
        on_error=_on_error,
    )


def run_real(
    *,
    run_dir: Path,
    resources: ResourceMapV1,
    overrides: list[str],
) -> Path:
    """Run real backend replay and record the real output directory in `run_state.json`."""
    run_dir = run_dir.expanduser().resolve()
    out_dir = run_dir / "real"

    def _on_error(run_state: dict[str, Any], ended_at: str) -> dict[str, Any]:
        backend_key = str((run_state.get("presets") or {}).get("backend", "unknown"))
        artifacts = dict(run_state.get("artifacts") or {})
        artifacts["real"] = {
            "real_run_dir": str(out_dir.resolve()),
            "backend": backend_key,
            "status": "failed",
            "ended_at": ended_at,
            "overrides": list(overrides),
        }
        run_state["artifacts"] = artifacts
        return run_state

    def _impl() -> Path:
        state = load_run_state(run_dir=run_dir)
        trace = require_trace_artifacts(run_dir=run_dir)

        cfg = _compose_stage_cfg(
            config_name="real_bench",
            allowed_groups={"model", "hardware", "backend", "workload"},
            state=state,
            resources=resources,
            overrides=overrides,
        )

        backend_name = str(cfg.backend.name)
        model_id = str(cfg.model.model_id)
        model_ref = Path(str(cfg.model.tokenizer_ref)).expanduser()
        device = str(cfg.hardware.device)
        sarathi_chunk_size: int | None = None
        sarathi_max_num_seqs: int | None = None
        sarathi_max_tokens: int | None = None
        sarathi_ignore_eos: bool | None = None
        if backend_name == "sarathi":
            scheduler = getattr(cfg.backend, "scheduler", None)
            if scheduler is not None:
                sarathi_chunk_size = int(getattr(scheduler, "chunk_size"))
                sarathi_max_num_seqs = int(getattr(scheduler, "max_num_seqs"))
            sarathi_max_tokens = int(getattr(cfg.backend, "max_tokens"))
            sarathi_ignore_eos = bool(getattr(cfg.backend, "ignore_eos"))

        out_dir.mkdir(parents=True, exist_ok=True)
        _ = run_token_length_replay(
            trace_csv=trace.trace_csv,
            backend=backend_name,
            model_id=model_id,
            model_ref=model_ref,
            device=device,
            out_dir=out_dir,
            sarathi_chunk_size=sarathi_chunk_size,
            sarathi_max_num_seqs=sarathi_max_num_seqs,
            sarathi_max_tokens=sarathi_max_tokens,
            sarathi_ignore_eos=sarathi_ignore_eos,
        )

        paper_fidelity_csv: Path | None = None
        if backend_name == "sarathi":
            candidate = out_dir / "paper_fidelity" / "request_metrics.csv"
            if candidate.exists():
                paper_fidelity_csv = candidate

        _update_run_state_real_ok(
            run_dir=run_dir,
            real_run_dir=out_dir,
            backend=backend_name,
            paper_fidelity_request_metrics_csv=paper_fidelity_csv,
            overrides=overrides,
        )
        return out_dir.resolve()

    return run_with_failure_json(
        run_dir=run_dir,
        stage="real",
        fn=_impl,
        failure_context={"run_dir": str(run_dir), "out_dir": str(out_dir)},
        on_error=_on_error,
    )


def run_report(
    *,
    run_dir: Path,
    overrides: list[str],
) -> Path:
    """Generate a sim-vs-real comparison report under `<run_dir>/report/`."""
    run_dir = run_dir.expanduser().resolve()
    report_dir = run_dir / "report"
    summary_md = report_dir / "summary.md"

    def _on_error(run_state: dict[str, Any], ended_at: str) -> dict[str, Any]:
        artifacts = dict(run_state.get("artifacts") or {})
        artifacts["report"] = {
            "report_dir": str(report_dir.resolve()),
            "summary_md": str(summary_md.resolve()),
            "status": "failed",
            "ended_at": ended_at,
            "overrides": list(overrides),
        }
        run_state["artifacts"] = artifacts
        return run_state

    def _impl() -> Path:
        _ = load_run_state(run_dir=run_dir)
        sim = require_sim_artifacts(run_dir=run_dir)
        real = require_real_artifacts(run_dir=run_dir)
        profile = require_profile_artifacts(run_dir=run_dir)

        from gpu_simulate_test.vidur_cli.reporting import write_paper_fidelity_style_report

        report_dir.mkdir(parents=True, exist_ok=True)
        _ = write_paper_fidelity_style_report(
            run_dir=run_dir,
            report_dir=report_dir,
            sim_run_dir=sim.sim_run_dir,
            real_run_dir=real.real_run_dir,
            profiling_root=profile.profiling_root,
            include_cpu_overhead=bool(profile.include_cpu_overhead),
        )

        _update_run_state_report_ok(run_dir=run_dir, report_dir=report_dir, summary_md=summary_md, overrides=overrides)
        return summary_md.resolve()

    return run_with_failure_json(
        run_dir=run_dir,
        stage="report",
        fn=_impl,
        failure_context={"run_dir": str(run_dir), "report_dir": str(report_dir)},
        on_error=_on_error,
    )


def _write_resolved_config_yaml(
    *,
    run_dir: Path,
    run_tag: str,
    presets: Presets,
    overrides: list[str],
    resources: ResourceMapV1,
) -> Path:
    """Write an optional run context snapshot to `<run_dir>/resolved_config.yaml`."""
    try:
        from omegaconf import OmegaConf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("omegaconf is required; run inside the Pixi env (`pixi install`).") from e

    payload = {
        "schema_version": "v1",
        "created_at": utcnow_iso(),
        "run_dir": str(run_dir.resolve()),
        "run_tag": str(run_tag),
        "presets": {
            "model": presets.model,
            "hardware": presets.hardware,
            "backend": presets.backend,
            "workload": presets.workload,
            "vidur": presets.vidur,
        },
        "overrides": list(overrides),
        "resources": dict(resources.to_json()),
    }
    sanitized = json.loads(json.dumps(payload, default=str))
    cfg = OmegaConf.create(sanitized)
    out = run_dir / "resolved_config.yaml"
    out.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return out.resolve()


def _load_workload_inputs(
    *,
    state: Mapping[str, Any],
    resources: ResourceMapV1,
    overrides: list[str],
) -> tuple[ArrivalScheduleConfig, Path, Path, int]:
    """Load workload inputs via Hydra composition (`workload_spec`)."""
    presets = state.get("presets") or {}
    model_key = str(presets.get("model"))
    workload_key = str(presets.get("workload"))

    stage_overrides = _filter_group_overrides(overrides, allowed_groups={"model", "workload"})
    injected = _resource_injection_overrides(resources)
    cfg = compose_config(
        config_name="workload_spec",
        config_roots=resources.hydra_config_roots,
        overrides=[*stage_overrides, f"model={model_key}", f"workload={workload_key}", *injected],
    )

    # Hydra returns an OmegaConf DictConfig; use attribute access defensively.
    kind = str(cfg.workload.arrival.kind)
    schedule = ArrivalScheduleConfig(
        kind=kind,
        seed=int(cfg.workload.arrival.seed),
        inter_arrival_ns=int(cfg.workload.arrival.inter_arrival_ns),
        poisson_rate_per_s=float(cfg.workload.arrival.poisson_rate_per_s),
    )
    prompts_jsonl = Path(str(cfg.workload.prompts)).expanduser()
    tokenizer_ref = Path(str(cfg.model.tokenizer_ref)).expanduser()
    return schedule, prompts_jsonl, tokenizer_ref, int(cfg.workload.num_decode_tokens)


def _compose_stage_cfg(
    *,
    config_name: str,
    allowed_groups: set[str],
    state: Mapping[str, Any],
    resources: ResourceMapV1,
    overrides: list[str],
) -> object:
    """Compose a stage config and force preset group selections from run_state.json."""
    presets = state.get("presets") or {}
    required_groups = {"model", "hardware", "backend", "workload", "vidur"} & set(allowed_groups)

    missing = [g for g in sorted(required_groups) if g not in presets]
    if missing:
        raise UserFacingError(
            "run_state.json is missing required preset selections.",
            hint="Recreate the run directory using `svr init-run`.",
            context={"missing": missing},
        )

    stage_overrides = _filter_group_overrides(overrides, allowed_groups=allowed_groups)
    group_overrides = [f"{g}={presets[g]}" for g in sorted(required_groups)]
    injected = _resource_injection_overrides(resources)
    return compose_config(
        config_name=config_name,
        config_roots=resources.hydra_config_roots,
        overrides=[*stage_overrides, *group_overrides, *injected],
    )


def _maybe_register_vidur_model(*, model_key: str, cfg: object) -> None:
    """Register any local Vidur model configs needed for profiling/simulation."""
    if model_key != "qwen3_0_6b":
        return
    try:
        tokenizer_ref = Path(str(cfg.model.tokenizer_ref)).expanduser()
    except Exception:
        return
    config_json = tokenizer_ref / "config.json"
    register_qwen3_0_6b(model_ref=Qwen3ModelRef(config_json=config_json))


def _filter_group_overrides(overrides: list[str], *, allowed_groups: set[str]) -> list[str]:
    out: list[str] = []
    for item in overrides:
        key = item.split("=", 1)[0]
        # Hydra supports leading "+" (append) and "~" (delete) operators.
        normalized_key = key.lstrip("+~")
        root = normalized_key.split(".", 1)[0]
        # Filter out any group-scoped overrides (including nested keys like `workload.*`)
        # when the group is not part of this stage's composed config.
        if root in {"model", "hardware", "backend", "workload", "vidur"} and root not in allowed_groups:
            continue
        out.append(item)
    return out


def _resource_injection_overrides(resources: ResourceMapV1) -> list[str]:
    repo_root = str(resources.repo_root.value)
    workspace_root = str(resources.workspace_root.value)
    models_root = str(resources.models_root.value)
    datasets_root = str(resources.datasets_root.value)
    return [
        f"paths.repo_root={repo_root}",
        f"paths.tmp_root={workspace_root}/tmp",
        f"+paths.models_root={models_root}",
        f"+paths.datasets_root={datasets_root}",
    ]


def _maybe_materialize_default_prompts(prompts_path: Path, *, resources: ResourceMapV1) -> Path:
    """Ensure the default prompts file exists under the workspace (best-effort copy)."""
    prompts_path = prompts_path.expanduser()
    if prompts_path.exists():
        return prompts_path.resolve()

    # Best-effort: if the user is using the repo default prompt path template, copy it.
    candidate = (
        resources.repo_root.value / "tmp" / "prompts" / "example.prompts.jsonl"
    ).expanduser().resolve()
    if candidate.exists():
        prompts_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, prompts_path)
        return prompts_path.resolve()

    raise UserFacingError(
        f"prompts file does not exist: {prompts_path}",
        hint="Provide a valid workload.prompts path via configs/overrides or use --from-lengths / --import-trace.",
        context={"expected_default_repo_prompt": str(candidate)},
    )


def _update_run_state_trace_ok(*, run_dir: Path, result: TraceBuildResult, overrides: list[str]) -> None:
    state = load_run_state(run_dir=run_dir)
    artifacts = dict(state.get("artifacts") or {})
    artifacts["trace"] = {
        "trace_csv": str(result.trace_csv),
        "trace_meta_json": str(result.trace_meta_json),
        "status": "ok",
        "ended_at": utcnow_iso(),
        "overrides": list(overrides),
        "trace_lengths_csv": str(result.trace_lengths_csv),
        "trace_intervals_csv": str(result.trace_intervals_csv),
    }
    state["artifacts"] = artifacts
    save_run_state(run_dir=run_dir, run_state=state)


def _update_run_state_profile_ok(
    *,
    run_dir: Path,
    profiling_root: Path,
    include_cpu_overhead: bool,
    overrides: list[str],
    mlp: dict[str, Any] | None = None,
) -> None:
    state = load_run_state(run_dir=run_dir)
    artifacts = dict(state.get("artifacts") or {})
    payload: dict[str, Any] = {
        "profiling_root": str(profiling_root.resolve()),
        "include_cpu_overhead": bool(include_cpu_overhead),
        "status": "ok",
        "ended_at": utcnow_iso(),
        "overrides": list(overrides),
    }
    if mlp is not None:
        payload["mlp"] = mlp
    artifacts["profile"] = payload
    state["artifacts"] = artifacts
    save_run_state(run_dir=run_dir, run_state=state)


def _update_run_state_sim_ok(
    *,
    run_dir: Path,
    sim_run_dir: Path,
    paper_fidelity_request_metrics_csv: Path | None,
    overrides: list[str],
) -> None:
    state = load_run_state(run_dir=run_dir)
    artifacts = dict(state.get("artifacts") or {})
    payload: dict[str, Any] = {
        "sim_run_dir": str(sim_run_dir.resolve()),
        "status": "ok",
        "ended_at": utcnow_iso(),
        "overrides": list(overrides),
    }
    if paper_fidelity_request_metrics_csv is not None:
        payload["paper_fidelity_request_metrics_csv"] = str(paper_fidelity_request_metrics_csv.resolve())
    artifacts["sim"] = payload
    state["artifacts"] = artifacts
    save_run_state(run_dir=run_dir, run_state=state)


def _update_run_state_real_ok(
    *,
    run_dir: Path,
    real_run_dir: Path,
    backend: str,
    paper_fidelity_request_metrics_csv: Path | None,
    overrides: list[str],
) -> None:
    state = load_run_state(run_dir=run_dir)
    artifacts = dict(state.get("artifacts") or {})
    payload: dict[str, Any] = {
        "real_run_dir": str(real_run_dir.resolve()),
        "backend": str(backend),
        "status": "ok",
        "ended_at": utcnow_iso(),
        "overrides": list(overrides),
    }
    if paper_fidelity_request_metrics_csv is not None:
        payload["paper_fidelity_request_metrics_csv"] = str(paper_fidelity_request_metrics_csv.resolve())
    artifacts["real"] = payload
    state["artifacts"] = artifacts
    save_run_state(run_dir=run_dir, run_state=state)


def _update_run_state_report_ok(
    *,
    run_dir: Path,
    report_dir: Path,
    summary_md: Path,
    overrides: list[str],
) -> None:
    state = load_run_state(run_dir=run_dir)
    artifacts = dict(state.get("artifacts") or {})
    artifacts["report"] = {
        "report_dir": str(report_dir.resolve()),
        "summary_md": str(summary_md.resolve()),
        "status": "ok",
        "ended_at": utcnow_iso(),
        "overrides": list(overrides),
    }
    state["artifacts"] = artifacts
    save_run_state(run_dir=run_dir, run_state=state)


def _read_arrival_kind(run_dir: Path) -> str:
    """Read the arrival schedule kind from `<run_dir>/trace/trace_meta.json`."""
    from gpu_simulate_test.io import read_json

    trace = require_trace_artifacts(run_dir=run_dir)
    meta = read_json(trace.trace_meta_json)
    arrival = meta.get("arrival_schedule", {}) if isinstance(meta, dict) else {}
    kind = arrival.get("kind") if isinstance(arrival, dict) else None
    return str(kind) if kind is not None else "unknown"


def _enrich_summary_md(summary_md: Path, *, arrival_kind: str, include_cpu_overhead: bool) -> None:
    """Ensure `summary.md` includes arrival kind and CPU overhead status."""
    content = summary_md.read_text(encoding="utf-8").splitlines()

    if any("arrival_kind:" in line for line in content):
        return

    insert_after = None
    for idx, line in enumerate(content):
        if line.startswith("- sim:"):
            insert_after = idx
            break

    extra: list[str] = [f"- arrival_kind: `{arrival_kind}`"]
    if include_cpu_overhead:
        extra.append("- cpu_overhead: enabled")
    else:
        extra.append("- cpu_overhead: disabled")
        extra.append("  - WARNING: CPU overhead was disabled during profiling; sim-vs-real parity may be affected.")

    if insert_after is None:
        updated = [*extra, "", *content]
    else:
        updated = [*content[: insert_after + 1], *extra, *content[insert_after + 1 :]]

    summary_md.write_text("\n".join(updated) + "\n", encoding="utf-8")


def _maybe_write_sim_paper_fidelity_metrics(*, sim_run_dir: Path) -> Path:
    """Materialize paper-fidelity-style `request_metrics.csv` for Vidur sim outputs.

    This keeps the canonical `sim/request_metrics.csv` (ttft-based schema) intact, while
    writing an additional CSV with Vidur's normalized paper-fidelity columns under:

        <sim_run_dir>/paper_fidelity/request_metrics.csv
    """
    from gpu_simulate_test.io import read_json, utcnow_iso, write_csv, write_json
    from gpu_simulate_test.paper_fidelity.scoring import PAPER_FIDELITY_REQUEST_METRICS_REQUIRED_COLUMNS
    from gpu_simulate_test.vidur_ext.sim_runner import convert_vidur_request_metrics_to_paper_fidelity

    meta = read_json(sim_run_dir / "run_meta.json")
    raw_dir_value = meta.get("vidur_raw_dir")
    if not isinstance(raw_dir_value, str) or not raw_dir_value:
        raise UserFacingError(
            "sim/run_meta.json is missing vidur_raw_dir; cannot build paper-fidelity request_metrics.csv.",
            hint="Re-run `vidur-cli svr sim --run-dir <run_dir>`.",
        )

    raw_dir = Path(raw_dir_value).expanduser()
    raw_csv = raw_dir / "request_metrics.csv"
    if not raw_csv.exists():
        raise UserFacingError(
            "Vidur raw request_metrics.csv is missing; cannot build paper-fidelity request_metrics.csv.",
            hint="Re-run `vidur-cli svr sim --run-dir <run_dir>`.",
            context={"raw_csv": str(raw_csv)},
        )

    df = convert_vidur_request_metrics_to_paper_fidelity(raw_csv)
    out_dir = sim_run_dir / "paper_fidelity"
    out_csv = out_dir / "request_metrics.csv"
    write_csv(out_csv, df, required_columns=PAPER_FIDELITY_REQUEST_METRICS_REQUIRED_COLUMNS)
    write_json(
        out_dir / "run_meta.json",
        {
            "schema_version": "v1",
            "run_type": "sim",
            "generated_at": utcnow_iso(),
            "raw_request_metrics_csv": str(raw_csv.resolve()),
            "request_metrics_csv": str(out_csv.resolve()),
        },
    )
    return out_csv.resolve()
