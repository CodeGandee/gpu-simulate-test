from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual smoke: compare-runs (CPU-only if inputs exist)")
    parser.add_argument("--real-run-dir", type=Path, required=True)
    parser.add_argument("--sim-run-dir", type=Path, required=True)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "gpu_simulate_test.cli.compare_runs",
        f"real_run_dir={args.real_run_dir}",
        f"sim_run_dir={args.sim_run_dir}",
    ]
    out = subprocess.check_output(cmd, cwd=_repo_root()).decode("utf-8").strip().splitlines()
    out_dir = Path(out[-1]).resolve()

    if not (out_dir / "summary.md").exists():
        raise SystemExit(f"Missing summary.md in {out_dir}")

    figs = list((out_dir / "figs").glob("*"))
    if not figs:
        raise SystemExit(f"Expected at least one plot under {out_dir / 'figs'}")

    print(f"OK: {out_dir}")


if __name__ == "__main__":
    main()

