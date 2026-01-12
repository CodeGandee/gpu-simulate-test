"""
Host profiling bundle exporter for Vidur microbenchmarks.

This module runs Vidur profiling entrypoints on the current machine (GPU required) and exports a
curated, Vidur-compatible profiling root under a user-provided output directory. By default, it
captures compute profiling (MLP + attention) and can optionally include CPU overhead profiling.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from gpu_simulate_test.io import build_env_snapshot, get_git_info, utcnow_iso, write_json
from gpu_simulate_test.vidur_ext.profile_runner import VidurProfileInputs, run_vidur_profiling


def _file_safe_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S-%f")


def _resolve_path(path: Path, *, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _ensure_clean_output_dir(*, output_dir: Path, cache_dir: Path) -> None:
    """Ensure `output_dir` does not already contain curated outputs.

    Hydra may create its run directory before user code runs. When `cache_dir` is inside
    `output_dir`, we allow `output_dir` to contain only the top-level directory segment(s)
    required for `cache_dir`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_top_level: set[str] = set()
    try:
        rel = cache_dir.resolve().relative_to(output_dir.resolve())
    except ValueError:
        rel = None
    if rel is not None and rel.parts:
        allowed_top_level.add(rel.parts[0])

    for child in output_dir.iterdir():
        if child.name in allowed_top_level:
            continue
        raise FileExistsError(
            f"output.dir must not contain curated outputs (refusing to overwrite): {output_dir}"
        )


def run_vidur_profiling_bundle(cfg: DictConfig, *, repo_root: Path) -> Path:
    """Run Vidur profiling and export a profiling bundle.

    Returns the created profiling root directory (cfg.output.dir).
    """
    started_at = utcnow_iso()
    run_id = _file_safe_timestamp()

    model_slug = str(OmegaConf.select(cfg, "output.model_slug") or "unknown")
    scheduler_name = str(OmegaConf.select(cfg, "output.scheduler_name") or "unknown")

    profiling_root = _resolve_path(Path(cfg.output.dir).expanduser(), repo_root=repo_root)
    cache_dir = _resolve_path(Path(cfg.output.cache_dir).expanduser(), repo_root=repo_root)

    _ensure_clean_output_dir(output_dir=profiling_root, cache_dir=cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_id = str(cfg.model.model_id)
    hardware_id = str(cfg.hardware.hardware_id)
    network_device = str(cfg.hardware.network_device)

    attention_backend_val = OmegaConf.select(cfg, "profiling.attention.backend")
    attention_backend = str(attention_backend_val) if attention_backend_val is not None else None

    cpu_overhead_validation_val = OmegaConf.select(cfg, "profiling.cpu_overhead.validation")
    cpu_overhead_validation = (
        str(cpu_overhead_validation_val).lower().strip()
        if cpu_overhead_validation_val is not None
        else "strict"
    )

    vidur_result = run_vidur_profiling(
        VidurProfileInputs(
            model_id=model_id,
            hardware_id=hardware_id,
            profiling_root=profiling_root,
            network_device=network_device,
            num_gpus=int(cfg.profiling.num_gpus),
            tensor_parallel_size=int(cfg.profiling.tensor_parallel_size),
            max_tokens=int(cfg.profiling.max_tokens),
            staging_root=cache_dir,
            include_network=bool(cfg.profiling.include_network),
            include_cpu_overhead=bool(cfg.profiling.cpu_overhead.enabled),
            cpu_overhead_max_batch_size=int(cfg.profiling.cpu_overhead.max_batch_size),
            cpu_overhead_validation=cpu_overhead_validation,
            attention_backend=attention_backend,
            attention_block_size=int(cfg.profiling.attention.block_size),
            attention_min_batch_size=int(cfg.profiling.attention.min_batch_size),
            attention_max_batch_size=int(cfg.profiling.attention.max_batch_size),
            attention_profile_mode=str(cfg.profiling.attention.profile_mode),
            allow_attention_fallback=bool(cfg.profiling.allow_attention_fallback),
        ),
        repo_root=repo_root,
    )

    git = get_git_info(repo_root=repo_root)
    params = OmegaConf.to_container(cfg, resolve=True)
    meta: dict[str, Any] = {
        "schema_version": "v1",
        "run_type": "vidur_profiling_bundle",
        "run_id": run_id,
        "model_id": model_id,
        "model_slug": model_slug,
        "scheduler_name": scheduler_name,
        "hardware_id": hardware_id,
        "network_device": network_device,
        "profiling_root": str(profiling_root),
        "cache_dir": str(cache_dir),
        "started_at": started_at,
        "ended_at": utcnow_iso(),
        "git_commit": git.commit or "unknown",
        "git_dirty": git.dirty,
        "env": build_env_snapshot(),
        "params": params if isinstance(params, dict) else None,
        "profiling_commands": {
            "mlp": vidur_result.mlp_cmd,
            "attention": vidur_result.attention_cmd,
            "cpu_overhead": vidur_result.cpu_overhead_cmd,
        },
        "profiling_outputs": {
            "mlp_csv": str(vidur_result.mlp_csv.resolve()),
            "attention_csv": str(vidur_result.attention_csv.resolve()),
            "attention_profiled": bool(vidur_result.attention_profiled),
            "cpu_overheads_csv": str(vidur_result.cpu_overheads_csv.resolve())
            if vidur_result.cpu_overheads_csv is not None
            else None,
            "cpu_overhead_profiled": bool(vidur_result.cpu_overhead_profiled),
        },
    }

    write_json(profiling_root / "profiling_meta.json", meta)
    return profiling_root
