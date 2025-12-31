from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from gpu_simulate_test.config import register_omegaconf_resolvers
from gpu_simulate_test.io import build_env_snapshot, get_git_info, utcnow_iso, write_json
from gpu_simulate_test.vidur_ext.profile_runner import VidurProfileInputs, run_vidur_profiling
from gpu_simulate_test.vidur_ext.qwen3_model_config import Qwen3ModelRef, register_qwen3_0_6b


register_omegaconf_resolvers()


@hydra.main(
    config_path="../../../configs/compare_vidur_real",
    config_name="vidur_profile",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    profiling_root = Path.cwd()
    repo_root = Path(cfg.paths.repo_root)

    # Ensure Vidur can resolve the model name via subclass discovery.
    register_qwen3_0_6b(
        model_ref=Qwen3ModelRef(
            config_json=repo_root / "models" / "qwen3-0.6b" / "source-data" / "config.json"
        )
    )

    started_at = utcnow_iso()
    git = get_git_info(repo_root=repo_root)

    inputs = VidurProfileInputs(
        model_id=str(cfg.model.model_id),
        hardware_id=str(cfg.hardware.hardware_id),
        profiling_root=profiling_root,
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

