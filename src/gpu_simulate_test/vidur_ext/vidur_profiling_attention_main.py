from __future__ import annotations

from pathlib import Path

from gpu_simulate_test.vidur_ext.qwen3_model_config import Qwen3ModelRef, register_qwen3_0_6b


def main() -> None:
    repo_root = Path.cwd()
    register_qwen3_0_6b(
        model_ref=Qwen3ModelRef(
            config_json=repo_root / "models" / "qwen3-0.6b" / "source-data" / "config.json"
        )
    )

    import vidur.profiling.attention.main as attention_main
    from vidur.profiling.utils import get_max_num_blocks as _get_max_num_blocks

    def _patched_get_max_num_blocks(*args, **kwargs):  # type: ignore[no-untyped-def]
        max_pp = int(kwargs.get("max_pipeline_parallel_size", 8))
        num_layers = int(args[0].num_layers)  # args[0] is a vidur.profiling.common.model_config.ModelConfig
        if num_layers % max_pp != 0:
            for candidate in [4, 2, 1]:
                if num_layers % candidate == 0:
                    kwargs["max_pipeline_parallel_size"] = candidate
                    break
        return _get_max_num_blocks(*args, **kwargs)

    attention_main.get_max_num_blocks = _patched_get_max_num_blocks  # type: ignore[assignment]
    attention_main.main()


if __name__ == "__main__":
    main()
