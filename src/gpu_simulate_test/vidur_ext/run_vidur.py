from __future__ import annotations

from pathlib import Path

from gpu_simulate_test.vidur_ext.qwen3_model_config import Qwen3ModelRef, register_qwen3_0_6b


def run_vidur_main(*, repo_root: Path) -> None:
    register_qwen3_0_6b(
        model_ref=Qwen3ModelRef(
            config_json=repo_root / "models" / "qwen3-0.6b" / "source-data" / "config.json"
        )
    )

    from vidur.main import main as vidur_main

    vidur_main()

