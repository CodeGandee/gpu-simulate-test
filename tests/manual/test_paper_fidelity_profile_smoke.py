"""
Manual smoke test for `paper-fidelity profile` (GPU required).

This script runs the profiling workflow for a scenario and verifies a Vidur-compatible profiling
root is produced under `tmp/paper_fidelity/profiling_roots/...`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual smoke: paper-fidelity profile (GPU required)")
    parser.add_argument("--scenario", type=str, default="llama2_7b_arxiv")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--device", type=str, default="a100")
    parser.add_argument("--network-device", type=str, default="a100_pairwise_nvlink")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--mlp-profile-method", type=str, default="cuda_event")
    parser.add_argument("--include-cpu-overhead", action="store_true")
    parser.add_argument("--cpu-overhead-max-batch-size", type=int, default=16)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "gpu_simulate_test.cli.paper_fidelity",
        "profile",
        "--scenario",
        args.scenario,
        f"profiling.max_tokens={args.max_tokens}",
        f"profiling.num_gpus={args.num_gpus}",
        f"profiling.mlp.profile_method={args.mlp_profile_method}",
    ]
    if args.include_cpu_overhead:
        cmd.append("--include-cpu-overhead")
        cmd.append(f"profiling.cpu_overhead.max_batch_size={args.cpu_overhead_max_batch_size}")
    out = subprocess.check_output(cmd, cwd=_repo_root()).decode("utf-8").strip().splitlines()
    profiling_root = Path(out[-1]).resolve()

    compute_dir = profiling_root / "data" / "profiling" / "compute" / args.device / Path(args.model_id)
    if not (compute_dir / "mlp.csv").exists() or not (compute_dir / "attention.csv").exists():
        raise SystemExit(f"Missing compute profiling under {compute_dir}")

    if args.include_cpu_overhead:
        cpu_dir = (
            profiling_root
            / "data"
            / "profiling"
            / "cpu_overhead"
            / args.network_device
            / Path(args.model_id)
        )
        cpu_csv = cpu_dir / "cpu_overheads.csv"
        if not cpu_csv.exists():
            raise SystemExit(f"Missing CPU overhead profiling under {cpu_csv}")
        if cpu_csv.stat().st_size <= 0:
            raise SystemExit(f"CPU overhead CSV is empty (size=0): {cpu_csv}")

    if not (profiling_root / "profiling_meta.json").exists():
        raise SystemExit(f"Missing profiling metadata under {profiling_root}")

    print(f"OK: {profiling_root}")


if __name__ == "__main__":
    main()
