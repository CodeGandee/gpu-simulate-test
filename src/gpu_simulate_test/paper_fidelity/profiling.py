"""
Host profiling orchestration for paper-fidelity “gap reproduction”.

This module generates a Vidur-compatible profiling root under `tmp/paper_fidelity/` by running
Vidur’s profiling entrypoints on the current machine (GPU required).
"""

from __future__ import annotations

import sys
import traceback as tb
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from omegaconf import OmegaConf

from gpu_simulate_test.env_guard import count_visible_gpus
from gpu_simulate_test.io import utcnow_iso, write_json
from gpu_simulate_test.paper_fidelity.failure_record import (
    build_failure_record,
    categorize_blocker,
    write_failure_record,
)
from gpu_simulate_test.paper_fidelity.paths import PaperFidelityPaths, build_run_meta
from gpu_simulate_test.paper_fidelity.validation import preflight_profile
from gpu_simulate_test.vidur_ext.profile_runner import VidurProfileInputs, VidurProfileResult, run_vidur_profiling


class PaperFidelityProfilingError(RuntimeError):
    def __init__(self, message: str, *, failure_record_path: Path) -> None:
        super().__init__(message)
        self.failure_record_path = failure_record_path


def _file_safe_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S-%f")


def _vidur_profile_result_jsonable(result: VidurProfileResult) -> dict[str, Any]:
    """Convert a `VidurProfileResult` to a JSON-friendly dict."""
    data = asdict(result)
    for key in ["profiling_root", "staging_root", "mlp_csv", "attention_csv", "cpu_overheads_csv"]:
        if key in data and data[key] is not None:
            data[key] = str(data[key])
    return data


def run_paper_fidelity_profiling(cfg: DictConfig, *, repo_root: Path) -> Path:
    """Run host profiling and return the created profiling root.

    Parameters
    ----------
    cfg
        Hydra config for the profiling run (scenario + profiling knobs).
    repo_root
        Repository root used to resolve scenario paths and locate the Vidur submodule.

    Returns
    -------
    pathlib.Path
        Absolute path to the created Vidur-compatible profiling root under
        `tmp/paper_fidelity/profiling_roots/<scenario>/<run_id>/`.
    """
    scenario_name = str(cfg.scenario.name)
    scenario_key = str(OmegaConf.select(cfg, "hydra.runtime.choices.scenario") or scenario_name)
    started_at = utcnow_iso()
    run_id = _file_safe_timestamp()

    pf_paths = PaperFidelityPaths(repo_root=repo_root)
    profiling_outputs_dir = pf_paths.profiling_outputs_dir(scenario_name, run_id)
    profiling_root = pf_paths.profiling_root_dir(scenario_name, run_id)
    profiling_meta_json = pf_paths.profiling_meta_path(scenario_name, run_id)
    failure_record_json = profiling_root / "failure_record.json"

    profiling_outputs_dir.mkdir(parents=True, exist_ok=True)
    profiling_root.mkdir(parents=True, exist_ok=True)

    try:
        available_gpus = count_visible_gpus()
        preflight_profile(cfg, repo_root=repo_root, available_gpus=available_gpus)

        model_id = str(cfg.scenario.model.model_id)
        device = str(cfg.scenario.vidur.device)
        network_device = str(cfg.scenario.vidur.network_device)

        num_gpus_val = OmegaConf.select(cfg, "profiling.num_gpus")
        num_gpus = int(num_gpus_val) if num_gpus_val is not None else 1

        max_tokens_val = OmegaConf.select(cfg, "profiling.max_tokens") or OmegaConf.select(
            cfg, "scenario.trace_source.max_tokens"
        )
        max_tokens = int(max_tokens_val) if max_tokens_val is not None else 4096

        tp_val = OmegaConf.select(cfg, "profiling.tensor_parallel_size") or OmegaConf.select(
            cfg, "scenario.vidur.tensor_parallel_size"
        )
        tensor_parallel_size = int(tp_val) if tp_val is not None else 1

        include_cpu_val = OmegaConf.select(cfg, "profiling.include_cpu_overhead")
        include_cpu_overhead = bool(include_cpu_val) if include_cpu_val is not None else False

        cpu_max_bs_val = OmegaConf.select(cfg, "profiling.cpu_overhead.max_batch_size")
        cpu_overhead_max_batch_size = int(cpu_max_bs_val) if cpu_max_bs_val is not None else 128

        cpu_validation_val = OmegaConf.select(cfg, "profiling.cpu_overhead.validation")
        cpu_overhead_validation = (
            str(cpu_validation_val).lower().strip() if cpu_validation_val is not None else "strict"
        )

        mlp_profile_method_val = OmegaConf.select(cfg, "profiling.mlp.profile_method")
        if mlp_profile_method_val is None:
            raise ValueError("profiling.mlp.profile_method is required (no default).")
        mlp_profile_method = str(mlp_profile_method_val).strip()
        normalized_mlp_profile_method = mlp_profile_method.lower()
        allowed_mlp_profile_methods = {
            "cuda_event",
            "record_function",
            "record_function_org",
            "kineto",
            "perf_counter",
        }
        if normalized_mlp_profile_method not in allowed_mlp_profile_methods:
            raise ValueError(
                f"Unsupported profiling.mlp.profile_method={mlp_profile_method!r}. "
                "Choose one of: cuda_event | record_function | record_function_org | kineto | perf_counter."
            )
        mlp_profile_method = normalized_mlp_profile_method

        mlp_validation_mode_val = OmegaConf.select(cfg, "profiling.mlp.validation.mode")
        mlp_validation_mode = str(mlp_validation_mode_val or "strict").lower().strip()
        if mlp_validation_mode not in {"strict", "non_strict"}:
            raise ValueError(
                f"profiling.mlp.validation.mode must be 'strict' or 'non_strict' (got {mlp_validation_mode!r})."
            )

        mlp_nan_policy_val = OmegaConf.select(cfg, "profiling.mlp.validation.nan_policy")
        mlp_nan_policy = str(mlp_nan_policy_val or "auto").lower().strip()
        if mlp_nan_policy not in {"auto", "reject", "drop", "zero"}:
            raise ValueError(
                "profiling.mlp.validation.nan_policy must be one of auto|reject|drop|zero "
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

        vidur_result = run_vidur_profiling(
            VidurProfileInputs(
                model_id=model_id,
                hardware_id=device,
                profiling_root=profiling_root,
                mlp_profile_method=mlp_profile_method,
                mlp_validation_mode=mlp_validation_mode,  # type: ignore[arg-type]
                mlp_nan_policy=mlp_nan_policy,  # type: ignore[arg-type]
                mlp_small_input_threshold=mlp_small_input_threshold,
                mlp_zero_heavy_limit=mlp_zero_heavy_limit,
                mlp_fallback_enabled=mlp_fallback_enabled,
                mlp_fallback_method=mlp_fallback_method,
                network_device=network_device,
                num_gpus=num_gpus,
                tensor_parallel_size=tensor_parallel_size,
                max_tokens=max_tokens,
                staging_root=profiling_outputs_dir,
                include_cpu_overhead=include_cpu_overhead,
                cpu_overhead_max_batch_size=cpu_overhead_max_batch_size,
                cpu_overhead_validation=cpu_overhead_validation,
                model_ref=Path(cfg.scenario.model.model_ref).expanduser(),
            ),
            repo_root=repo_root,
        )

        meta_params = OmegaConf.to_container(cfg, resolve=True)
        extra: dict[str, Any] = {
            "profiling": {
                "device": device,
                "network_device": network_device,
                "model_id": model_id,
                "num_gpus": num_gpus,
                "tensor_parallel_size": tensor_parallel_size,
                "max_tokens": max_tokens,
                "include_cpu_overhead": include_cpu_overhead,
                "cpu_overhead_max_batch_size": cpu_overhead_max_batch_size,
                "cpu_overhead_validation": cpu_overhead_validation,
                "mlp_nan_policy": mlp_nan_policy,
            },
            "profiling_commands": {"mlp": vidur_result.mlp_cmd, "attention": vidur_result.attention_cmd},
            "profiling_outputs": {
                "mlp_csv": str(vidur_result.mlp_csv.resolve()),
                "attention_csv": str(vidur_result.attention_csv.resolve()),
                "attention_profiled": bool(vidur_result.attention_profiled),
                "cpu_overheads_csv": str(vidur_result.cpu_overheads_csv.resolve())
                if vidur_result.cpu_overheads_csv is not None
                else None,
                "cpu_overhead_profiled": bool(vidur_result.cpu_overhead_profiled),
                "cpu_overhead_validation": vidur_result.extra.get("cpu_overhead_validation"),
            },
            "vidur_profile_result": _vidur_profile_result_jsonable(vidur_result),
        }

        meta = build_run_meta(
            repo_root=repo_root,
            run_type="paper_fidelity_profile",
            run_id=run_id,
            scenario_name=scenario_name,
            started_at=started_at,
            ended_at=utcnow_iso(),
            params=meta_params if isinstance(meta_params, dict) else None,
            artifacts={
                "profiling_root": profiling_root,
                "profiling_outputs_dir": profiling_outputs_dir,
                "profiling_meta_json": profiling_meta_json,
            },
            extra=extra,
        )
        meta["mlp_profile_method"] = vidur_result.extra.get("mlp_profile_method")
        meta["mlp_vidur_profile_method"] = vidur_result.extra.get("mlp_vidur_profile_method")
        meta["mlp_fallback"] = vidur_result.extra.get("mlp_fallback")
        meta["mlp_validation"] = vidur_result.extra.get("mlp_validation")
        write_json(profiling_meta_json, meta)
        return profiling_root.resolve()
    except PaperFidelityProfilingError:
        raise
    except Exception as e:
        stack = tb.format_exc()
        category = categorize_blocker(error_message=f"{type(e).__name__}: {e}", traceback=stack)
        record = build_failure_record(
            run_id=run_id,
            action="profile",
            scenario_key=scenario_key,
            scenario_name=scenario_name,
            workload=None,
            scale=None,
            attempted_command=list(sys.argv),
            hydra_overrides=[],
            error_message=f"{type(e).__name__}: {e}",
            traceback=stack,
            blocker_category=category,
        )
        failure_record_path = write_failure_record(failure_record_json, record)
        raise PaperFidelityProfilingError(
            f"{type(e).__name__}: {e}",
            failure_record_path=failure_record_path,
        ) from e
