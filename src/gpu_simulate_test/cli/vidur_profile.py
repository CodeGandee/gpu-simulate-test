from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from gpu_simulate_test.config import register_omegaconf_resolvers
from gpu_simulate_test.io import build_env_snapshot, get_git_info, utcnow_iso, write_json
from gpu_simulate_test.vidur_ext.profile_runner import VidurProfileInputs, run_vidur_profiling
from gpu_simulate_test.vidur_ext.qwen3_model_config import maybe_register_qwen3_0_6b


register_omegaconf_resolvers()


@hydra.main(
    config_path="../../../configs/compare_vidur_real",
    config_name="vidur_profile",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    profiling_root = Path.cwd()
    repo_root = Path(cfg.paths.repo_root)

    model_id = str(cfg.model.model_id)
    tokenizer_ref = Path(str(cfg.model.tokenizer_ref)).expanduser()
    maybe_register_qwen3_0_6b(model_id=model_id, tokenizer_ref=tokenizer_ref)

    started_at = utcnow_iso()
    git = get_git_info(repo_root=repo_root)

    num_gpus_val = OmegaConf.select(cfg, "profiling.num_gpus")
    num_gpus = int(num_gpus_val or 1)
    if num_gpus < 1:
        raise ValueError(f"profiling.num_gpus must be >= 1 (got {num_gpus!r}).")

    tensor_parallel_size_val = OmegaConf.select(cfg, "profiling.tensor_parallel_size")
    tensor_parallel_size = int(tensor_parallel_size_val or 1)
    if tensor_parallel_size < 1:
        raise ValueError(f"profiling.tensor_parallel_size must be >= 1 (got {tensor_parallel_size!r}).")

    max_tokens_val = OmegaConf.select(cfg, "profiling.max_tokens")
    max_tokens = int(max_tokens_val or 4096)
    if max_tokens < 1:
        raise ValueError(f"profiling.max_tokens must be >= 1 (got {max_tokens!r}).")

    include_network_val = OmegaConf.select(cfg, "profiling.include_network")
    include_network = bool(include_network_val) if include_network_val is not None else True

    network_device_val = OmegaConf.select(cfg, "hardware.network_device")
    if network_device_val is None:
        raise ValueError("hardware.network_device is required for profiling (no default).")
    network_device = str(network_device_val)

    cpu_overhead_enabled_val = OmegaConf.select(cfg, "profiling.cpu_overhead.enabled")
    include_cpu_overhead = bool(cpu_overhead_enabled_val) if cpu_overhead_enabled_val is not None else True

    cpu_overhead_max_batch_size_val = OmegaConf.select(cfg, "profiling.cpu_overhead.max_batch_size")
    cpu_overhead_max_batch_size = int(cpu_overhead_max_batch_size_val or 128)
    if cpu_overhead_max_batch_size < 1:
        raise ValueError(
            "profiling.cpu_overhead.max_batch_size must be >= 1 "
            f"(got {cpu_overhead_max_batch_size!r})."
        )

    cpu_overhead_validation_val = OmegaConf.select(cfg, "profiling.cpu_overhead.validation")
    cpu_overhead_validation = str(cpu_overhead_validation_val or "strict").lower().strip()
    if cpu_overhead_validation not in {"strict", "warn", "off"}:
        raise ValueError(
            "profiling.cpu_overhead.validation must be one of strict|warn|off "
            f"(got {cpu_overhead_validation!r})."
        )

    attention_profile_mode_val = OmegaConf.select(cfg, "profiling.attention.profile_mode")
    attention_profile_mode = str(attention_profile_mode_val or "both").lower().strip()
    if attention_profile_mode not in {"decode", "prefill", "both"}:
        raise ValueError(
            "profiling.attention.profile_mode must be one of decode|prefill|both "
            f"(got {attention_profile_mode!r})."
        )

    attention_backend_val = OmegaConf.select(cfg, "profiling.attention.backend")
    attention_backend = str(attention_backend_val).strip() if attention_backend_val is not None else None
    if attention_backend == "":
        attention_backend = None

    attention_block_size_val = OmegaConf.select(cfg, "profiling.attention.block_size")
    attention_block_size = int(attention_block_size_val or 16)
    if attention_block_size < 1:
        raise ValueError(f"profiling.attention.block_size must be >= 1 (got {attention_block_size!r}).")

    attention_min_batch_size_val = OmegaConf.select(cfg, "profiling.attention.min_batch_size")
    attention_min_batch_size = int(attention_min_batch_size_val or 1)
    if attention_min_batch_size < 1:
        raise ValueError(
            "profiling.attention.min_batch_size must be >= 1 "
            f"(got {attention_min_batch_size!r})."
        )

    attention_max_batch_size_val = OmegaConf.select(cfg, "profiling.attention.max_batch_size")
    attention_max_batch_size = int(attention_max_batch_size_val or 1)
    if attention_max_batch_size < attention_min_batch_size:
        raise ValueError(
            "profiling.attention.max_batch_size must be >= profiling.attention.min_batch_size "
            f"(got min={attention_min_batch_size!r} max={attention_max_batch_size!r})."
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

    mlp_small_input_threshold_val = OmegaConf.select(cfg, "profiling.mlp.validation.small_input_threshold")
    mlp_small_input_threshold = int(mlp_small_input_threshold_val or 128)

    mlp_zero_heavy_limit_val = OmegaConf.select(cfg, "profiling.mlp.validation.zero_heavy_limit")
    mlp_zero_heavy_limit = float(mlp_zero_heavy_limit_val or 0.01)

    mlp_fallback_enabled_val = OmegaConf.select(cfg, "profiling.mlp.fallback.enabled")
    mlp_fallback_enabled = bool(mlp_fallback_enabled_val or False)

    mlp_fallback_method_val = OmegaConf.select(cfg, "profiling.mlp.fallback.method")
    mlp_fallback_method = str(mlp_fallback_method_val or "cuda_event").strip()

    inputs = VidurProfileInputs(
        model_id=model_id,
        hardware_id=str(cfg.hardware.hardware_id),
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
        include_network=include_network,
        include_cpu_overhead=include_cpu_overhead,
        cpu_overhead_max_batch_size=cpu_overhead_max_batch_size,
        cpu_overhead_validation=cpu_overhead_validation,
        attention_backend=attention_backend,
        attention_block_size=attention_block_size,
        attention_min_batch_size=attention_min_batch_size,
        attention_max_batch_size=attention_max_batch_size,
        attention_profile_mode=attention_profile_mode,
    )
    run_vidur_profiling(inputs, repo_root=repo_root)

    run_meta = {
        "schema_version": "v1",
        "run_type": "vidur_profile",
        "run_id": str(cfg.output.run_id),
        "model": str(cfg.model.model_id),
        "profiling_root": str(profiling_root.resolve()),
        "hardware": OmegaConf.to_container(cfg.hardware, resolve=True),
        "started_at": started_at,
        "ended_at": utcnow_iso(),
        "git_commit": git.commit or "unknown",
        "git_dirty": git.dirty,
        "env": build_env_snapshot(),
        "params": OmegaConf.to_container(cfg, resolve=True),
        "hydra": {"config_path": "configs/compare_vidur_real", "config_name": "vidur_profile"},
    }
    write_json(profiling_root / "run_meta.json", run_meta)
    print(str(profiling_root))


if __name__ == "__main__":
    main()
