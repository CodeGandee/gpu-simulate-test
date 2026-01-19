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

    mlp_profile_method_val = OmegaConf.select(cfg, "profiling.mlp.profile_method")
    if mlp_profile_method_val is None:
        raise ValueError("profiling.mlp.profile_method is required (no default).")
    mlp_profile_method = str(mlp_profile_method_val).strip()

    mlp_validation_mode_val = OmegaConf.select(cfg, "profiling.mlp.validation.mode")
    mlp_validation_mode = str(mlp_validation_mode_val or "strict").lower().strip()
    if mlp_validation_mode not in {"strict", "non_strict"}:
        raise ValueError(
            f"profiling.mlp.validation.mode must be 'strict' or 'non_strict' (got {mlp_validation_mode!r})."
        )

    mlp_nan_policy_val = OmegaConf.select(cfg, "profiling.mlp.validation.nan_policy")
    mlp_nan_policy = str(mlp_nan_policy_val or "auto").lower().strip()
    if mlp_nan_policy not in {"auto", "reject", "drop"}:
        raise ValueError(
            "profiling.mlp.validation.nan_policy must be one of auto|reject|drop "
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
