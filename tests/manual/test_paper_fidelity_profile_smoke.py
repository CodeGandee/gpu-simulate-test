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
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--num-gpus", type=int, default=1)
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
    ]
    out = subprocess.check_output(cmd, cwd=_repo_root()).decode("utf-8").strip().splitlines()
    profiling_root = Path(out[-1]).resolve()

    compute_dir = profiling_root / "data" / "profiling" / "compute" / args.device / Path(args.model_id)
    if not (compute_dir / "mlp.csv").exists() or not (compute_dir / "attention.csv").exists():
        raise SystemExit(f"Missing compute profiling under {compute_dir}")

    if not (profiling_root / "profiling_meta.json").exists():
        raise SystemExit(f"Missing profiling metadata under {profiling_root}")

    print(f"OK: {profiling_root}")


if __name__ == "__main__":
    main()
