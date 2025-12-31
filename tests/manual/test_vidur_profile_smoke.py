from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual smoke: vidur-profile (A100 required)")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--profiling-root", type=Path, required=True)
    parser.add_argument("--hardware-id", type=str, default="a100")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "gpu_simulate_test.cli.vidur_profile",
        f"model.model_id={args.model_id}",
        f"hardware.hardware_id={args.hardware_id}",
        f"vidur.profiling.root={args.profiling_root}",
    ]
    out = subprocess.check_output(cmd, cwd=_repo_root()).decode("utf-8").strip().splitlines()
    root = Path(out[-1]).resolve()

    compute_dir = root / "data" / "profiling" / "compute" / args.hardware_id / Path(args.model_id)
    if not (compute_dir / "mlp.csv").exists() or not (compute_dir / "attention.csv").exists():
        raise SystemExit(f"Missing compute profiling under {compute_dir}")

    print(f"OK: {root}")


if __name__ == "__main__":
    main()

