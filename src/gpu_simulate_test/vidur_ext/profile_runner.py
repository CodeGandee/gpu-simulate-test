"""
Vidur profiling runner for host-calibrated paper-fidelity workflows.

This module wraps Vidur's profiling entrypoints (MLP, attention, and optionally CPU overhead) to produce a
Vidur-compatible profiling root (`data/profiling/...`). Callers typically store large, intermediate
artifacts under `tmp/` and keep the final profiling root stable for reuse in simulations.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    include_network
        Whether to stage Vidur network profiling CSVs into the profiling root when available.
    include_cpu_overhead
        Whether to run Vidur's CPU overhead profiler and stage its CSV into the profiling root.
        Disabled by default to match the Vidur paper's evaluation practice (optimized serving stack
        to eliminate unnecessary CPU overheads).
    cpu_overhead_max_batch_size
        Maximum batch size to profile in the CPU overhead profiler. Vidur profiles a fixed grid of
        batch sizes up to this value.
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
    allow_attention_fallback
        Whether to fall back to a packaged template `attention.csv` when attention profiling fails.
    """

    model_id: str
    hardware_id: str
    profiling_root: Path
    network_device: str = "a100_pairwise_nvlink"
    num_gpus: int = 1
    tensor_parallel_size: int = 1
    max_tokens: int = 4096
    staging_root: Path | None = None
    include_network: bool = True
    include_cpu_overhead: bool = False
    cpu_overhead_max_batch_size: int = 128
    attention_backend: str | None = None
    attention_block_size: int = 16
    attention_min_batch_size: int = 1
    attention_max_batch_size: int = 1
    attention_profile_mode: str = "decode"
    allow_attention_fallback: bool = True


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


def _pick_attention_template(*, repo_root: Path, hardware_id: str) -> Path:
    """Select a packaged attention profiling template CSV for a given hardware id.

    This is used as a fallback when the attention profiling entrypoint fails, so that the
    simulator can still run with a minimal attention CSV that matches the target model.
    """
    candidates = [
        repo_root
        / "extern"
        / "tracked"
        / "vidur"
        / "data"
        / "profiling"
        / "compute"
        / hardware_id
        / "microsoft"
        / "phi-2"
        / "attention.csv",
        repo_root
        / "extern"
        / "tracked"
        / "vidur"
        / "data"
        / "profiling"
        / "compute"
        / hardware_id
        / "meta-llama"
        / "Llama-2-7b-hf"
        / "attention.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No attention.csv template found for hardware_id={hardware_id} under {candidates[0].parents[5]}")


def _write_attention_fallback(
    *,
    template_csv: Path,
    out_csv: Path,
    model_id: str,
    tensor_parallel_size: int,
    block_size: int,
) -> None:
    """Write a minimal attention profiling CSV derived from a packaged template.

    The template rows are adjusted to match the target model and tensor-parallel setting so that
    Vidur's predictor can filter the rows correctly.
    """
    import pandas as pd
    from vidur.config.model_config import BaseModelConfig

    model_cfg = BaseModelConfig.create_from_name(model_id)
    df = pd.read_csv(template_csv).drop_duplicates()

    df["n_embd"] = int(model_cfg.embedding_dim)
    df["n_q_head"] = int(model_cfg.num_q_heads)
    df["n_kv_head"] = int(model_cfg.num_kv_heads)
    df["block_size"] = int(block_size)
    df["num_tensor_parallel_workers"] = int(tensor_parallel_size)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


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
        If the MLP profiling entrypoint fails. (Attention profiling failures fall back to a
        packaged template.)
    """
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

    compute_ready = mlp_dst.exists() and attn_dst.exists()
    cpu_ready = (not inputs.include_cpu_overhead) or (
        cpu_overheads_dst is not None and cpu_overheads_dst.exists()
    )
    if compute_ready and cpu_ready:
        return VidurProfileResult(
            profiling_root=inputs.profiling_root,
            staging_root=staging,
            mlp_csv=mlp_dst,
            attention_csv=attn_dst,
            attention_profiled=False,
            cpu_overheads_csv=cpu_overheads_dst if inputs.include_cpu_overhead else None,
            cpu_overhead_profiled=False,
            mlp_cmd=[],
            attention_cmd=[],
            cpu_overhead_cmd=[],
            extra={"skipped": True},
        )

    mlp_cmd: list[str] = []
    attn_cmd: list[str] = []
    attention_profiled = False
    extra: dict[str, Any] = {}
    import pandas as pd

    if not compute_ready:
        mlp_cmd = [
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
        ]
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

        subprocess.check_call(mlp_cmd, cwd=repo_root)
        attn_exc: subprocess.CalledProcessError | None = None
        try:
            subprocess.check_call(attn_cmd, cwd=repo_root)
            attention_ok = True
        except subprocess.CalledProcessError as e:
            attention_ok = False
            attn_exc = e

        mlp_latest = _latest_dir(staging / "mlp")
        mlp_src = mlp_latest / inputs.model_id / "mlp.csv"
        compute_dst_dir.mkdir(parents=True, exist_ok=True)

        mlp_df = pd.read_csv(mlp_src).drop_duplicates()
        time_cols = [c for c in mlp_df.columns if c.startswith("time_stats.")]
        mlp_df[time_cols] = mlp_df[time_cols].fillna(0.0)
        mlp_df.to_csv(mlp_dst, index=False)

        if attention_ok:
            attn_latest = _latest_dir(staging / "attention")
            attn_src = attn_latest / inputs.model_id / "attention.csv"
            shutil.copy2(attn_src, attn_dst)
            attention_profiled = True
        elif not inputs.allow_attention_fallback:
            raise attn_exc if attn_exc is not None else subprocess.CalledProcessError(
                returncode=1, cmd=attn_cmd
            )
        else:
            template = _pick_attention_template(
                repo_root=repo_root, hardware_id=inputs.hardware_id
            )
            _write_attention_fallback(
                template_csv=template,
                out_csv=attn_dst,
                model_id=inputs.model_id,
                tensor_parallel_size=int(inputs.tensor_parallel_size),
                block_size=int(inputs.attention_block_size),
            )
            extra["attention_fallback_template"] = str(template)
    else:
        extra["skipped_compute"] = True

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
            subprocess.check_call(cpu_overhead_cmd, cwd=repo_root)
            cpu_latest = _latest_dir(staging / "cpu_overhead")
            cpu_src = cpu_latest / inputs.model_id / "cpu_overhead.csv"
            cpu_overheads_dst.parent.mkdir(parents=True, exist_ok=True)
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
