from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from gpu_simulate_test.config import register_omegaconf_resolvers
from gpu_simulate_test.io import build_env_snapshot, get_git_info, utcnow_iso
from gpu_simulate_test.vidur_ext.qwen3_model_config import Qwen3ModelRef, register_qwen3_0_6b
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

    register_qwen3_0_6b(
        model_ref=Qwen3ModelRef(
            config_json=repo_root / "models" / "qwen3-0.6b" / "source-data" / "config.json"
        )
    )

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

    inputs = VidurSimInputs(
        workload_dir=workload_dir,
        profiling_root=profiling_root,
        model_id=str(cfg.model.model_id),
        device=str(cfg.hardware.hardware_id),
    )
    run_vidur_sim(inputs, out_dir=out_dir, run_meta=run_meta)
    print(str(out_dir))


if __name__ == "__main__":
    main()
