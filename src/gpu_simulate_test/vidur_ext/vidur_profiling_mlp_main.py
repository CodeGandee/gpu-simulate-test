from __future__ import annotations

import os
import sys
from pathlib import Path

from gpu_simulate_test.vidur_ext.qwen3_model_config import Qwen3ModelRef, register_qwen3_0_6b


def _parse_profile_method(argv: list[str]) -> str | None:
    for idx, arg in enumerate(argv):
        if arg == "--profile_method" and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith("--profile_method="):
            return arg.split("=", 1)[1]
    return None


def _maybe_patch_record_function_tracer(argv: list[str]) -> None:
    method = (_parse_profile_method(argv) or "").strip().lower()
    if method != "record_function":
        return

    os.environ["GPU_SIMULATE_TEST_ENABLE_VIDUR_MLP_RECORD_FUNCTION_TRACER_V2"] = "1"
    try:
        from gpu_simulate_test.vidur_ext.record_function_tracer_v2 import RecordFunctionTracerV2
        import vidur.profiling.mlp.mlp_wrapper as mlp_wrapper

        mlp_wrapper.RecordFunctionTracer = RecordFunctionTracerV2  # type: ignore[assignment]
    except Exception as e:
        raise RuntimeError(
            "Failed to patch Vidur MLP record-function tracer for driver-launched kernels."
        ) from e


def main() -> None:
    repo_root = Path.cwd()
    from gpu_simulate_test.env_guard import (
        apply_cuda_visible_devices_from_gsim,
        patch_sarathi_preserve_cuda_visible_devices,
    )

    apply_cuda_visible_devices_from_gsim(repo_root=repo_root)
    patch_sarathi_preserve_cuda_visible_devices()

    register_qwen3_0_6b(
        model_ref=Qwen3ModelRef(
            config_json=repo_root / "models" / "qwen3-0.6b" / "source-data" / "config.json"
        )
    )

    _maybe_patch_record_function_tracer(sys.argv[1:])

    from vidur.profiling.mlp.main import main as vidur_mlp_main

    vidur_mlp_main()


if __name__ == "__main__":
    main()
