from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from gpu_simulate_test.config import register_omegaconf_resolvers
from gpu_simulate_test.vidur_ext.profiling_bundle import run_vidur_profiling_bundle


register_omegaconf_resolvers()


@hydra.main(
    config_path="../../../configs/vidur_profiling",
    config_name="bundle",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    repo_root = Path(cfg.paths.repo_root)
    profiling_root = run_vidur_profiling_bundle(cfg, repo_root=repo_root)
    print(str(profiling_root))


if __name__ == "__main__":
    main()

