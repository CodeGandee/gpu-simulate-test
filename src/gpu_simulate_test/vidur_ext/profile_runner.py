"""
Vidur profiling runner for host-calibrated paper-fidelity workflows.

This module wraps Vidur's profiling entrypoints (MLP, attention, and optionally CPU overhead) to produce a
Vidur-compatible profiling root (`data/profiling/...`). Callers typically store large, intermediate
artifacts under `tmp/` and keep the final profiling root stable for reuse in simulations.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

MlpValidationMode = Literal["strict", "non_strict"]
MlpNanPolicy = Literal["auto", "reject", "drop", "zero"]


@dataclass(frozen=True)
class VidurProfileInputs:
    """Inputs for running Vidur profiling entrypoints.

    Parameters
    ----------
    model_id
        HuggingFace model id used by Vidur, e.g. `meta-llama/Llama-2-7b-hf`.
    hardware_id
        Vidur hardware id (used in profiling directory layout), e.g. `a100`.
    profiling_root
        Output root directory to populate with `data/profiling/...`.
    network_device
        Vidur network device id, e.g. `a100_pairwise_nvlink`.
    num_gpus
        Number of GPUs to use for profiling.
    tensor_parallel_size
        Tensor-parallel degree (Vidur uses this to filter profiling rows).
    max_tokens
        Maximum sequence length for profiling runs.
    staging_root
        Optional directory for large intermediate profiler outputs; defaults to
        `<profiling_root>/_staging` when omitted.
    mlp_profile_method
        REQUIRED. Profiling method passed to Vidur's MLP profiler (no hidden defaults).
        Examples: `record_function`, `cuda_event`.
        Repo-specific: `record_function_org` matches upstream Vidur's tracer behavior (no local patching),
        while still invoking Vidur with `record_function`.
    mlp_validation_mode
        Validation mode to apply when staging `mlp.csv` for zero-heavy checks:
        - `strict` (default) fails on zero-heavy signals.
        - `non_strict` warns on zero-heavy signals.
        Missing-value handling is controlled by `mlp_nan_policy`.
    mlp_nan_policy
        How to handle missing (NaN) core timing targets in `mlp.csv`.
        - auto (default): strict => reject; non_strict => drop (per-target) during consumption.
        - reject: always reject NaNs (ignores strict/non_strict for NaN handling).
        - drop: allow NaNs (consumers must drop missing samples per target before training).
        - zero: allow NaNs (consumers must fill missing targets with 0.0 per target before training).
    mlp_small_input_threshold
        Token count threshold below which exact zeros are tolerated in `mlp.csv`.
    mlp_zero_heavy_limit
        Fraction of exact zeros above which a column is flagged as "zero-heavy" for rows where
        `num_tokens >= mlp_small_input_threshold`.
    mlp_fallback_enabled
        Whether to automatically retry MLP profiling with `mlp_fallback_method` when validation
        fails.
    mlp_fallback_method
        Alternate profiling method to try when `mlp_fallback_enabled` is set.
    include_network
        Whether to stage Vidur network profiling CSVs into the profiling root when available.
    include_cpu_overhead
        Whether to run Vidur's CPU overhead profiler and stage its CSV into the profiling root.
        Disabled by default to match the Vidur paper's evaluation practice (optimized serving stack
        to eliminate unnecessary CPU overheads).
    cpu_overhead_max_batch_size
        Maximum batch size to profile in the CPU overhead profiler. Vidur profiles a fixed grid of
        batch sizes up to this value.
    cpu_overhead_validation
        Validation mode for staged `cpu_overheads.csv`: `strict` rejects placeholder-like dummy
        inputs; `warn` allows them but emits warnings; `off` disables placeholder checks (but still
        requires a parseable, non-empty CSV).
    attention_backend
        Optional attention backend passed to Vidur's attention profiler (Sarathi backend name).
        When unset, Vidur's default is used.
    attention_block_size
        Block size used for paged attention profiling.
    attention_min_batch_size
        Minimum decode batch size to profile in the attention profiler.
    attention_max_batch_size
        Maximum decode batch size to profile in the attention profiler.
    attention_profile_mode
        Which phase(s) to profile in the attention profiler: `decode`, `prefill`, or `both`.
    """

    model_id: str
    hardware_id: str
    profiling_root: Path
    mlp_profile_method: str
    mlp_validation_mode: MlpValidationMode = "strict"
    mlp_nan_policy: MlpNanPolicy = "auto"
    mlp_small_input_threshold: int = 128
    mlp_zero_heavy_limit: float = 0.01
    mlp_fallback_enabled: bool = False
    mlp_fallback_method: str = "cuda_event"
    network_device: str = "a100_pairwise_nvlink"
    num_gpus: int = 1
    tensor_parallel_size: int = 1
    max_tokens: int = 4096
    staging_root: Path | None = None
    include_network: bool = True
    include_cpu_overhead: bool = False
    cpu_overhead_max_batch_size: int = 128
    cpu_overhead_validation: str = "strict"
    attention_backend: str | None = None
    attention_block_size: int = 16
    attention_min_batch_size: int = 1
    attention_max_batch_size: int = 1
    attention_profile_mode: str = "both"
    model_ref: Path | None = None


@dataclass(frozen=True)
class VidurProfileResult:
    """Outputs of a profiling run.

    Attributes
    ----------
    profiling_root
        Root directory populated with the Vidur-compatible layout.
    staging_root
        Directory containing intermediate profiling outputs produced by Vidur.
    mlp_csv
        Final MLP profiling CSV staged under `data/profiling/compute/...`.
    attention_csv
        Final attention profiling CSV staged under `data/profiling/compute/...`.
    attention_profiled
        Whether attention profiling completed successfully on this host.
    cpu_overheads_csv
        Final CPU overhead profiling CSV staged under `data/profiling/cpu_overhead/...`, when
        enabled.
    cpu_overhead_profiled
        Whether CPU overhead profiling completed successfully on this host.
    mlp_cmd
        Command used to run the MLP profiler (for provenance).
    attention_cmd
        Command used to run the attention profiler (for provenance).
    cpu_overhead_cmd
        Command used to run the CPU overhead profiler (for provenance).
    extra
        Additional provenance or fallback details.
    """

    profiling_root: Path
    staging_root: Path
    mlp_csv: Path
    attention_csv: Path
    attention_profiled: bool
    cpu_overheads_csv: Path | None
    cpu_overhead_profiled: bool
    mlp_cmd: list[str]
    attention_cmd: list[str]
    cpu_overhead_cmd: list[str]
    extra: dict[str, Any]


def _copy_if_missing(src: Path, dst: Path) -> None:
    """Copy `src` to `dst` if `dst` does not already exist."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _latest_dir(base: Path) -> Path:
    """Return the most recently modified child directory under `base`."""
    candidates = [p for p in base.glob("*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No timestamped output dirs under {base}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _vidur_profile_method(*, requested_profile_method: str) -> str:
    """Map wrapper-level MLP profiling method names to Vidur-native values."""
    normalized = str(requested_profile_method).strip().lower()
    if normalized == "record_function_org":
        return "record_function"
    return str(requested_profile_method)


def run_vidur_profiling(inputs: VidurProfileInputs, *, repo_root: Path) -> VidurProfileResult:
    """Run Vidur profiling entrypoints and stage outputs into a profiling root.

    Parameters
    ----------
    inputs
        Profiling inputs describing the model, hardware, and output paths.
    repo_root
        Repository root used as the subprocess working directory and to locate Vidur data.

    Returns
    -------
    VidurProfileResult
        Paths to the staged profiling CSVs plus provenance information.

    Raises
    ------
    RuntimeError
        If `torch` is unavailable or CUDA is not accessible on this host.
    subprocess.CalledProcessError
        If any profiling entrypoint fails.
    """
    from gpu_simulate_test.env_guard import (
        apply_cuda_visible_devices_from_gsim,
        patch_sarathi_preserve_cuda_visible_devices,
    )

    apply_cuda_visible_devices_from_gsim(repo_root=repo_root)
    patch_sarathi_preserve_cuda_visible_devices()

    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is required; run inside the Pixi env (`pixi install`).") from e

    if not torch.cuda.is_available():
        raise RuntimeError("Vidur profiling requires a CUDA-capable GPU (torch.cuda.is_available() is False).")

    if inputs.attention_min_batch_size < 1 or inputs.attention_max_batch_size < 1:
        raise ValueError("attention_min_batch_size and attention_max_batch_size must both be >= 1")
    if inputs.attention_min_batch_size > inputs.attention_max_batch_size:
        raise ValueError("attention_min_batch_size must be <= attention_max_batch_size")

    inputs.profiling_root.mkdir(parents=True, exist_ok=True)

    profiling_base = inputs.profiling_root / "data" / "profiling"

    if inputs.include_network:
        vidur_data = repo_root / "extern" / "tracked" / "vidur" / "data" / "profiling"
        network_src = vidur_data / "network" / inputs.network_device
        if network_src.exists():
            _copy_if_missing(
                network_src / "all_reduce.csv",
                profiling_base / "network" / inputs.network_device / "all_reduce.csv",
            )
            _copy_if_missing(
                network_src / "send_recv.csv",
                profiling_base / "network" / inputs.network_device / "send_recv.csv",
            )

    if inputs.staging_root is None:
        staging = inputs.profiling_root / "_staging"
    else:
        staging = inputs.staging_root
    staging.mkdir(parents=True, exist_ok=True)

    def _subprocess_env(*, enable_attention_compat: bool) -> dict[str, str]:
        env = dict(os.environ)
        env["GPU_SIMULATE_TEST_ENABLE_VIDUR_ATTENTION_COMPAT"] = "1" if enable_attention_compat else "0"
        return env

    compute_dst_dir = profiling_base / "compute" / inputs.hardware_id / inputs.model_id
    mlp_dst = compute_dst_dir / "mlp.csv"
    attn_dst = compute_dst_dir / "attention.csv"
    cpu_overheads_dst: Path | None = None
    if inputs.include_cpu_overhead:
        cpu_overheads_dst = (
            profiling_base
            / "cpu_overhead"
            / inputs.network_device
            / inputs.model_id
            / "cpu_overheads.csv"
        )

    from gpu_simulate_test.vidur_ext.mlp_validation import MlpCsvValidationError, validate_mlp_csv

    def _validate_staged_mlp() -> dict[str, Any]:
        result = validate_mlp_csv(
            mlp_dst,
            mode=str(inputs.mlp_validation_mode).lower().strip() or "strict",  # type: ignore[arg-type]
            nan_policy=str(inputs.mlp_nan_policy).lower().strip() or "auto",  # type: ignore[arg-type]
            small_input_threshold=int(inputs.mlp_small_input_threshold),
            zero_heavy_limit=float(inputs.mlp_zero_heavy_limit),
        )
        if result.warnings:
            import warnings

            for warning in result.warnings:
                warnings.warn(warning)
        return result.as_jsonable()

    def _run_and_stage_mlp(*, profile_method: str) -> list[str]:
        import pandas as pd

        cmd = [
            sys.executable,
            "-m",
            "gpu_simulate_test.vidur_ext.vidur_profiling_mlp_main",
            "--num_gpus",
            str(int(inputs.num_gpus)),
            "--num_tensor_parallel_workers",
            str(int(inputs.tensor_parallel_size)),
            "--models",
            inputs.model_id,
            "--output_dir",
            str(staging),
            "--max_tokens",
            str(int(inputs.max_tokens)),
            "--profile_method",
            str(profile_method),
        ]
        subprocess.check_call(cmd, cwd=repo_root, env=_subprocess_env(enable_attention_compat=False))

        mlp_latest = _latest_dir(staging / "mlp")
        mlp_src = mlp_latest / inputs.model_id / "mlp.csv"
        compute_dst_dir.mkdir(parents=True, exist_ok=True)

        mlp_df = pd.read_csv(mlp_src).drop_duplicates()
        mlp_df.to_csv(mlp_dst, index=False)
        return cmd

    compute_ready = mlp_dst.exists() and attn_dst.exists()
    cpu_ready = (not inputs.include_cpu_overhead) or (
        cpu_overheads_dst is not None and cpu_overheads_dst.exists()
    )
    if compute_ready and cpu_ready:
        mlp_cmd: list[str] = []
        extra: dict[str, Any] = {
            "skipped": True,
            "mlp_profile_method": str(inputs.mlp_profile_method),
            "mlp_vidur_profile_method": _vidur_profile_method(
                requested_profile_method=str(inputs.mlp_profile_method)
            ),
            "mlp_fallback": {
                "enabled": bool(inputs.mlp_fallback_enabled),
                "used": False,
                "method": str(inputs.mlp_fallback_method),
            },
        }

        try:
            extra["mlp_validation"] = _validate_staged_mlp()
        except MlpCsvValidationError as e:
            extra["mlp_validation"] = e.result.as_jsonable()
            if not inputs.mlp_fallback_enabled:
                raise
            mlp_cmd = _run_and_stage_mlp(profile_method=str(inputs.mlp_fallback_method))
            extra["mlp_profile_method"] = str(inputs.mlp_fallback_method)
            extra["mlp_vidur_profile_method"] = _vidur_profile_method(
                requested_profile_method=str(inputs.mlp_fallback_method)
            )
            extra["mlp_fallback"]["used"] = True
            extra["mlp_validation"] = _validate_staged_mlp()

        if inputs.include_cpu_overhead and cpu_overheads_dst is not None and cpu_overheads_dst.exists():
            from gpu_simulate_test.vidur_ext.cpu_overhead_validation import validate_cpu_overheads_csv

            result = validate_cpu_overheads_csv(
                cpu_overheads_dst,
                mode=str(inputs.cpu_overhead_validation).lower().strip() or "strict",  # type: ignore[arg-type]
                expected_model_id=str(inputs.model_id),
                expected_tensor_parallel_degree=int(inputs.tensor_parallel_size),
            )
            extra["cpu_overhead_validation"] = result.as_jsonable()
        return VidurProfileResult(
            profiling_root=inputs.profiling_root,
            staging_root=staging,
            mlp_csv=mlp_dst,
            attention_csv=attn_dst,
            attention_profiled=False,
            cpu_overheads_csv=cpu_overheads_dst if inputs.include_cpu_overhead else None,
            cpu_overhead_profiled=False,
            mlp_cmd=mlp_cmd,
            attention_cmd=[],
            cpu_overhead_cmd=[],
            extra=extra,
        )

    mlp_cmd: list[str] = []
    attn_cmd: list[str] = []
    attention_profiled = False
    extra: dict[str, Any] = {}
    import pandas as pd

    if not compute_ready:
        extra["mlp_profile_method"] = str(inputs.mlp_profile_method)
        extra["mlp_vidur_profile_method"] = _vidur_profile_method(
            requested_profile_method=str(inputs.mlp_profile_method)
        )
        extra["mlp_fallback"] = {
            "enabled": bool(inputs.mlp_fallback_enabled),
            "used": False,
            "method": str(inputs.mlp_fallback_method),
        }

        mlp_cmd = _run_and_stage_mlp(profile_method=str(inputs.mlp_profile_method))
        attn_cmd = [
            sys.executable,
            "-m",
            "gpu_simulate_test.vidur_ext.vidur_profiling_attention_main",
            "--num_gpus",
            str(int(inputs.num_gpus)),
            "--num_tensor_parallel_workers",
            str(int(inputs.tensor_parallel_size)),
            "--models",
            inputs.model_id,
            "--output_dir",
            str(staging),
            "--max_model_len",
            str(int(inputs.max_tokens)),
            "--max_seq_len",
            str(int(inputs.max_tokens)),
            "--min_batch_size",
            str(int(inputs.attention_min_batch_size)),
            "--max_batch_size",
            str(int(inputs.attention_max_batch_size)),
        ]
        if inputs.attention_backend is not None:
            attn_cmd.extend(["--attention_backend", str(inputs.attention_backend)])
        attn_cmd.extend(["--block_size", str(int(inputs.attention_block_size))])

        mode = str(inputs.attention_profile_mode).lower().strip()
        if mode not in {"decode", "prefill", "both"}:
            raise ValueError(
                f"Unsupported attention_profile_mode={inputs.attention_profile_mode!r} (expected decode|prefill|both)"
            )
        if mode == "decode":
            attn_cmd.append("--profile_only_decode")
        elif mode == "prefill":
            attn_cmd.append("--profile_only_prefill")

        subprocess.check_call(attn_cmd, cwd=repo_root, env=_subprocess_env(enable_attention_compat=True))
        attn_latest = _latest_dir(staging / "attention")
        attn_src = attn_latest / inputs.model_id / "attention.csv"
        shutil.copy2(attn_src, attn_dst)
        attention_profiled = True

        try:
            extra["mlp_validation"] = _validate_staged_mlp()
        except MlpCsvValidationError as e:
            extra["mlp_validation"] = e.result.as_jsonable()
            if not inputs.mlp_fallback_enabled:
                raise
            mlp_cmd = _run_and_stage_mlp(profile_method=str(inputs.mlp_fallback_method))
            extra["mlp_profile_method"] = str(inputs.mlp_fallback_method)
            extra["mlp_vidur_profile_method"] = _vidur_profile_method(
                requested_profile_method=str(inputs.mlp_fallback_method)
            )
            extra["mlp_fallback"]["used"] = True
            extra["mlp_validation"] = _validate_staged_mlp()
    else:
        extra["skipped_compute"] = True
        extra["mlp_profile_method"] = str(inputs.mlp_profile_method)
        extra["mlp_vidur_profile_method"] = _vidur_profile_method(
            requested_profile_method=str(inputs.mlp_profile_method)
        )
        extra["mlp_fallback"] = {
            "enabled": bool(inputs.mlp_fallback_enabled),
            "used": False,
            "method": str(inputs.mlp_fallback_method),
        }
        extra["mlp_validation"] = _validate_staged_mlp()

    cpu_overhead_cmd: list[str] = []
    cpu_overhead_profiled = False
    if inputs.include_cpu_overhead and cpu_overheads_dst is not None:
        if cpu_overheads_dst.exists():
            extra["skipped_cpu_overhead"] = True
        else:
            cpu_overhead_cmd = [
                sys.executable,
                "-m",
                "gpu_simulate_test.vidur_ext.vidur_profiling_cpu_overhead_main",
                "--models",
                inputs.model_id,
                "--num_tensor_parallel_workers",
                str(int(inputs.tensor_parallel_size)),
                "--max_batch_size",
                str(int(inputs.cpu_overhead_max_batch_size)),
                "--output_dir",
                str(staging),
            ]
            if inputs.model_ref is not None and inputs.model_ref.exists():
                cpu_overhead_cmd.extend(["--model_path", str(inputs.model_ref.resolve())])

            subprocess.check_call(
                cpu_overhead_cmd,
                cwd=repo_root,
                env=_subprocess_env(enable_attention_compat=False),
            )
            cpu_latest = _latest_dir(staging / "cpu_overhead")
            cpu_src = cpu_latest / inputs.model_id / "cpu_overhead.csv"
            cpu_overheads_dst.parent.mkdir(parents=True, exist_ok=True)
            from gpu_simulate_test.vidur_ext.cpu_overhead_validation import validate_cpu_overheads_csv

            validation = validate_cpu_overheads_csv(
                cpu_src,
                mode=str(inputs.cpu_overhead_validation).lower().strip() or "strict",  # type: ignore[arg-type]
                expected_model_id=str(inputs.model_id),
                expected_tensor_parallel_degree=int(inputs.tensor_parallel_size),
            )
            extra["cpu_overhead_validation"] = validation.as_jsonable()
            cpu_df = pd.read_csv(cpu_src).drop_duplicates()
            cpu_df.to_csv(cpu_overheads_dst, index=False)
            cpu_overhead_profiled = True

    return VidurProfileResult(
        profiling_root=inputs.profiling_root,
        staging_root=staging,
        mlp_csv=mlp_dst,
        attention_csv=attn_dst,
        attention_profiled=attention_profiled,
        cpu_overheads_csv=cpu_overheads_dst if inputs.include_cpu_overhead else None,
        cpu_overhead_profiled=cpu_overhead_profiled,
        mlp_cmd=mlp_cmd,
        attention_cmd=attn_cmd,
        cpu_overhead_cmd=cpu_overhead_cmd,
        extra=extra,
    )
