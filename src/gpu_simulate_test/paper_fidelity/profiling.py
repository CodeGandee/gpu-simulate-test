"""
Host profiling orchestration for paper-fidelity “gap reproduction”.

This module generates a Vidur-compatible profiling root under `tmp/paper_fidelity/` by running
Vidur’s profiling entrypoints on the current machine (GPU required).
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from omegaconf import OmegaConf

from gpu_simulate_test.io import utcnow_iso, write_json
from gpu_simulate_test.paper_fidelity.paths import PaperFidelityPaths, build_run_meta
from gpu_simulate_test.vidur_ext.profile_runner import VidurProfileInputs, VidurProfileResult, run_vidur_profiling


def _file_safe_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S-%f")


def _vidur_profile_result_jsonable(result: VidurProfileResult) -> dict[str, Any]:
    """Convert a `VidurProfileResult` to a JSON-friendly dict."""
    data = asdict(result)
    for key in ["profiling_root", "staging_root", "mlp_csv", "attention_csv"]:
        if key in data:
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
    started_at = utcnow_iso()
    run_id = _file_safe_timestamp()

    pf_paths = PaperFidelityPaths(repo_root=repo_root)
    profiling_outputs_dir = pf_paths.profiling_outputs_dir(scenario_name, run_id)
    profiling_root = pf_paths.profiling_root_dir(scenario_name, run_id)
    profiling_meta_json = pf_paths.profiling_meta_path(scenario_name, run_id)

    profiling_outputs_dir.mkdir(parents=True, exist_ok=True)
    profiling_root.mkdir(parents=True, exist_ok=True)

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

    vidur_result = run_vidur_profiling(
        VidurProfileInputs(
            model_id=model_id,
            hardware_id=device,
            profiling_root=profiling_root,
            network_device=network_device,
            num_gpus=num_gpus,
            tensor_parallel_size=tensor_parallel_size,
            max_tokens=max_tokens,
            staging_root=profiling_outputs_dir,
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
        },
        "profiling_commands": {"mlp": vidur_result.mlp_cmd, "attention": vidur_result.attention_cmd},
        "profiling_outputs": {
            "mlp_csv": str(vidur_result.mlp_csv.resolve()),
            "attention_csv": str(vidur_result.attention_csv.resolve()),
            "attention_profiled": bool(vidur_result.attention_profiled),
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
    write_json(profiling_meta_json, meta)
    return profiling_root.resolve()
