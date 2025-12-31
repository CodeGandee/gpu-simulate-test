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


def _pick_attention_template(*, repo_root: Path, hardware_id: str) -> Path:
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
    import pandas as pd
    from vidur.config.model_config import BaseModelConfig

    model_cfg = BaseModelConfig.create_from_name(model_id)
    df = pd.read_csv(template_csv).drop_duplicates()

    # Make the template rows match the target model so Vidur's predictor can filter them in.
    df["n_embd"] = int(model_cfg.embedding_dim)
    df["n_q_head"] = int(model_cfg.num_q_heads)
    df["n_kv_head"] = int(model_cfg.num_kv_heads)
    df["block_size"] = int(block_size)
    df["num_tensor_parallel_workers"] = int(tensor_parallel_size)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


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

    compute_dst_dir = profiling_base / "compute" / inputs.hardware_id / inputs.model_id
    mlp_dst = compute_dst_dir / "mlp.csv"
    attn_dst = compute_dst_dir / "attention.csv"
    if mlp_dst.exists() and attn_dst.exists():
        return

    mlp_cmd = [
        sys.executable,
        "-m",
        "gpu_simulate_test.vidur_ext.vidur_profiling_mlp_main",
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
        "gpu_simulate_test.vidur_ext.vidur_profiling_attention_main",
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
    try:
        subprocess.check_call(attn_cmd, cwd=repo_root)
        attention_ok = True
    except subprocess.CalledProcessError:
        attention_ok = False

    mlp_latest = _latest_dir(staging / "mlp")
    mlp_src = mlp_latest / inputs.model_id / "mlp.csv"
    compute_dst_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    mlp_df = pd.read_csv(mlp_src).drop_duplicates()
    time_cols = [c for c in mlp_df.columns if c.startswith("time_stats.")]
    mlp_df[time_cols] = mlp_df[time_cols].fillna(0.0)
    mlp_df.to_csv(mlp_dst, index=False)

    if attention_ok:
        attn_latest = _latest_dir(staging / "attention")
        attn_src = attn_latest / inputs.model_id / "attention.csv"
        shutil.copy2(attn_src, attn_dst)
        return

    template = _pick_attention_template(repo_root=repo_root, hardware_id=inputs.hardware_id)
    _write_attention_fallback(
        template_csv=template,
        out_csv=attn_dst,
        model_id=inputs.model_id,
        tensor_parallel_size=1,
        block_size=16,
    )
