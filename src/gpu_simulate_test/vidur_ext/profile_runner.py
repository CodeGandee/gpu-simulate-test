from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VidurProfileInputs:
    model_id: str
    hardware_id: str
    profiling_root: Path
    network_device: str = "a100_pairwise_nvlink"


def _copy_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _latest_dir(base: Path) -> Path:
    candidates = [p for p in base.glob("*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No timestamped output dirs under {base}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_vidur_profiling(inputs: VidurProfileInputs, *, repo_root: Path) -> None:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is required; run inside the Pixi env (`pixi install`).") from e

    if not torch.cuda.is_available():
        raise RuntimeError("Vidur profiling requires a CUDA-capable GPU (torch.cuda.is_available() is False).")

    inputs.profiling_root.mkdir(parents=True, exist_ok=True)

    profiling_base = inputs.profiling_root / "data" / "profiling"

    # Stage network profiling from the Vidur submodule data (covers TP/PP>1 if needed).
    vidur_data = repo_root / "extern" / "tracked" / "vidur" / "data" / "profiling"
    network_src = vidur_data / "network" / inputs.network_device
    if network_src.exists():
        _copy_if_missing(network_src / "all_reduce.csv", profiling_base / "network" / inputs.network_device / "all_reduce.csv")
        _copy_if_missing(network_src / "send_recv.csv", profiling_base / "network" / inputs.network_device / "send_recv.csv")

    # Compute profiling: call Vidur profiling entrypoints (GPU-required).
    staging = inputs.profiling_root / "_staging"
    staging.mkdir(parents=True, exist_ok=True)

    mlp_cmd = [
        sys.executable,
        "-m",
        "vidur.profiling.mlp.main",
        "--num_gpus",
        "1",
        "--num_tensor_parallel_workers",
        "1",
        "--models",
        inputs.model_id,
        "--output_dir",
        str(staging),
        "--max_tokens",
        "4096",
    ]
    attn_cmd = [
        sys.executable,
        "-m",
        "vidur.profiling.attention.main",
        "--num_gpus",
        "1",
        "--num_tensor_parallel_workers",
        "1",
        "--models",
        inputs.model_id,
        "--output_dir",
        str(staging),
        "--max_model_len",
        "4096",
        "--max_seq_len",
        "4096",
        "--min_batch_size",
        "1",
        "--max_batch_size",
        "1",
        "--profile_only_decode",
    ]

    subprocess.check_call(mlp_cmd, cwd=repo_root)
    subprocess.check_call(attn_cmd, cwd=repo_root)

    mlp_latest = _latest_dir(staging / "mlp")
    attn_latest = _latest_dir(staging / "attention")

    mlp_src = mlp_latest / inputs.model_id / "mlp.csv"
    attn_src = attn_latest / inputs.model_id / "attention.csv"

    compute_dst_dir = profiling_base / "compute" / inputs.hardware_id / inputs.model_id
    compute_dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mlp_src, compute_dst_dir / "mlp.csv")
    shutil.copy2(attn_src, compute_dst_dir / "attention.csv")
