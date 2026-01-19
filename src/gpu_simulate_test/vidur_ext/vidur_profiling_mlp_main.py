from __future__ import annotations

import os
import sys
from pathlib import Path

from gpu_simulate_test.vidur_ext.qwen3_model_config import maybe_register_qwen3_0_6b


def _parse_profile_method(argv: list[str]) -> str | None:
    for idx, arg in enumerate(argv):
        if arg == "--profile_method" and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith("--profile_method="):
            return arg.split("=", 1)[1]
    return None


def _rewrite_profile_method_for_vidur(argv: list[str]) -> list[str]:
    """Rewrite wrapper-only profile_method values to Vidur-native values.

    This repo supports a wrapper-level alias `record_function_org` which means:
    - use upstream Vidur's record_function tracer (no local patching)
    - but pass `record_function` to Vidur (since that is what Vidur understands)
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--profile_method" and i + 1 < len(argv):
            value = str(argv[i + 1])
            normalized = value.strip().lower()
            if normalized == "record_function_org":
                out.extend([arg, "record_function"])
            else:
                out.extend([arg, value])
            i += 2
            continue
        if arg.startswith("--profile_method="):
            key, value = arg.split("=", 1)
            normalized = value.strip().lower()
            if normalized == "record_function_org":
                out.append(f"{key}=record_function")
            else:
                out.append(arg)
            i += 1
            continue
        out.append(arg)
        i += 1
    return out


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

    for model_id in models:
        maybe_register_qwen3_0_6b(
            model_id=model_id,
            tokenizer_ref=repo_root / "models" / "qwen3-0.6b" / "source-data",
        )


def _maybe_patch_record_function_tracer(*, requested_profile_method: str) -> None:
    method = requested_profile_method.strip().lower()
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

    requested_profile_method = _parse_profile_method(sys.argv[1:]) or ""

    _maybe_register_local_model_configs(repo_root=repo_root, argv=sys.argv[1:])

    _maybe_patch_record_function_tracer(requested_profile_method=requested_profile_method)

    sys.argv[1:] = _rewrite_profile_method_for_vidur(sys.argv[1:])

    from vidur.profiling.mlp.main import main as vidur_mlp_main

    vidur_mlp_main()


if __name__ == "__main__":
    main()
