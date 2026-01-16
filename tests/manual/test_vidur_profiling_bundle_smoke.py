from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual smoke: vidur-profiling bundle export (GPU required)")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--hardware-id", type=str, default="a100")
    parser.add_argument("--scheduler-name", type=str, default="sarathi-serve")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--mlp-profile-method", type=str, default="cuda_event")
    parser.add_argument("--attention-mode", type=str, default="both", choices=["decode", "prefill", "both"])
    parser.add_argument("--max-batch-size", type=int, default=1)
    args = parser.parse_args()

    repo_root = _repo_root()
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S-%f")
    output_dir = (
        repo_root
        / "tmp"
        / "vidur_profiling_bundle_smoke"
        / args.scheduler_name
        / run_id
    )

    cmd = [
        sys.executable,
        "-m",
        "gpu_simulate_test.cli.vidur_profiling_bundle",
        f"output.dir={output_dir}",
        f"output.model_slug=llama2-7b",
        f"output.scheduler_name={args.scheduler_name}",
        f"model.model_id={args.model_id}",
        f"hardware.hardware_id={args.hardware_id}",
        f"profiling.max_tokens={args.max_tokens}",
        f"profiling.num_gpus={args.num_gpus}",
        f"profiling.mlp.profile_method={args.mlp_profile_method}",
        f"profiling.attention.profile_mode={args.attention_mode}",
        f"profiling.attention.max_batch_size={args.max_batch_size}",
    ]
    out = subprocess.check_output(cmd, cwd=repo_root).decode("utf-8").strip().splitlines()
    profiling_root = Path(out[-1]).resolve()

    compute_dir = profiling_root / "data" / "profiling" / "compute" / args.hardware_id / Path(args.model_id)
    if not (compute_dir / "mlp.csv").exists() or not (compute_dir / "attention.csv").exists():
        raise SystemExit(f"Missing compute profiling under {compute_dir}")

    if not (profiling_root / "profiling_meta.json").exists():
        raise SystemExit(f"Missing profiling metadata under {profiling_root}")

    print(f"OK: {profiling_root}")


if __name__ == "__main__":
    main()
