from __future__ import annotations

import sys
from pathlib import Path

from gpu_simulate_test.vidur_ext.qwen3_model_config import maybe_register_qwen3_0_6b


def main() -> None:
    repo_root = Path.cwd()
    from gpu_simulate_test.env_guard import (
        apply_cuda_visible_devices_from_gsim,
        patch_sarathi_preserve_cuda_visible_devices,
    )

    apply_cuda_visible_devices_from_gsim(repo_root=repo_root)
    patch_sarathi_preserve_cuda_visible_devices()

    _maybe_register_local_model_configs(repo_root=repo_root, argv=sys.argv[1:])

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


def _parse_models(argv: list[str]) -> list[str]:
    out: list[str] = []
    for idx, arg in enumerate(argv):
        if arg == "--models":
            j = idx + 1
            while j < len(argv) and not str(argv[j]).startswith("-"):
                out.append(str(argv[j]))
                j += 1
            break
        if arg.startswith("--models="):
            out.extend(arg.split("=", 1)[1].split(","))
            break
    return [m for m in (str(x).strip() for x in out) if m]


def _maybe_register_local_model_configs(*, repo_root: Path, argv: list[str]) -> None:
    models = _parse_models(argv)
    if not models:
        return

    # For profiling entrypoints, `tokenizer_ref` is not passed explicitly; use repo-local model
    # refs when needed for local Vidur config registration.
    for model_id in models:
        maybe_register_qwen3_0_6b(
            model_id=model_id,
            tokenizer_ref=repo_root / "models" / "qwen3-0.6b" / "source-data",
        )


if __name__ == "__main__":
    main()
