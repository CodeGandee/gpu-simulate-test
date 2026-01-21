from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from gpu_simulate_test.config import register_omegaconf_resolvers
from gpu_simulate_test.io import build_env_snapshot, get_git_info, utcnow_iso
from gpu_simulate_test.vidur_ext.qwen3_model_config import maybe_register_qwen3_0_6b
from gpu_simulate_test.vidur_ext.sim_runner import VidurSimInputs, run_vidur_sim


register_omegaconf_resolvers()


@hydra.main(
    config_path="../../../configs/compare_vidur_real",
    config_name="vidur_sim",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    out_dir = Path.cwd()
    repo_root = Path(cfg.paths.repo_root)

    model_id = str(cfg.model.model_id)
    tokenizer_ref = Path(str(cfg.model.tokenizer_ref)).expanduser()
    maybe_register_qwen3_0_6b(model_id=model_id, tokenizer_ref=tokenizer_ref)

    workload_dir = Path(cfg.workload.workload_dir).expanduser()
    profiling_root = Path(cfg.vidur.profiling.root).expanduser()

    started_at = utcnow_iso()
    git = get_git_info(repo_root=repo_root)

    run_meta = {
        "schema_version": "v1",
        "run_type": "vidur",
        "run_id": str(cfg.output.run_id),
        "model": str(cfg.model.model_id),
        "workload_dir": str(workload_dir.resolve()),
        "profiling_root": str(profiling_root.resolve()),
        "hardware": OmegaConf.to_container(cfg.hardware, resolve=True),
        "started_at": started_at,
        "git_commit": git.commit or "unknown",
        "git_dirty": git.dirty,
        "env": build_env_snapshot(),
        "params": OmegaConf.to_container(cfg, resolve=True),
        "hydra": {"config_path": "configs/compare_vidur_real", "config_name": "vidur_sim"},
    }

    mlp_validation_mode_val = OmegaConf.select(cfg, "vidur.validation.mlp.mode")
    mlp_validation_mode = str(mlp_validation_mode_val or "strict").lower().strip()
    if mlp_validation_mode not in {"strict", "non_strict"}:
        raise ValueError(
            f"vidur.validation.mlp.mode must be 'strict' or 'non_strict' (got {mlp_validation_mode!r})."
        )

    mlp_nan_policy_val = OmegaConf.select(cfg, "vidur.validation.mlp.nan_policy")
    mlp_nan_policy = str(mlp_nan_policy_val or "auto").lower().strip()
    if mlp_nan_policy not in {"auto", "reject", "drop", "zero"}:
        raise ValueError(
            "vidur.validation.mlp.nan_policy must be one of auto|reject|drop|zero "
            f"(got {mlp_nan_policy!r})."
        )

    mlp_small_input_threshold_val = OmegaConf.select(cfg, "vidur.validation.mlp.small_input_threshold")
    mlp_small_input_threshold = int(mlp_small_input_threshold_val or 128)

    mlp_zero_heavy_limit_val = OmegaConf.select(cfg, "vidur.validation.mlp.zero_heavy_limit")
    mlp_zero_heavy_limit = float(mlp_zero_heavy_limit_val or 0.01)

    inputs = VidurSimInputs(
        workload_dir=workload_dir,
        profiling_root=profiling_root,
        model_id=model_id,
        device=str(cfg.hardware.hardware_id),
        mlp_validation_mode=mlp_validation_mode,
        mlp_nan_policy=mlp_nan_policy,
        mlp_small_input_threshold=mlp_small_input_threshold,
        mlp_zero_heavy_limit=mlp_zero_heavy_limit,
    )
    run_vidur_sim(inputs, out_dir=out_dir, run_meta=run_meta)
    print(str(out_dir))


if __name__ == "__main__":
    main()
