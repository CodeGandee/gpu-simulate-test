from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual smoke: workload-spec")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-decode-tokens", type=int, default=16)
    parser.add_argument("--kind", choices=["fixed_interval", "poisson"], default="fixed_interval")
    parser.add_argument("--inter-arrival-ns", type=int, default=0)
    parser.add_argument("--poisson-rate-per-s", type=float, default=2.0)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "gpu_simulate_test.cli.workload_spec",
        f"workload.prompts={args.prompts}",
        f"workload.seed={args.seed}",
        f"workload.num_decode_tokens={args.num_decode_tokens}",
        f"workload.arrival.kind={args.kind}",
        f"workload.arrival.inter_arrival_ns={args.inter_arrival_ns}",
        f"workload.arrival.poisson_rate_per_s={args.poisson_rate_per_s}",
    ]
    out = subprocess.check_output(cmd, cwd=_repo_root()).decode("utf-8").strip().splitlines()
    out_dir = Path(out[-1]).resolve()

    expected = [
        out_dir / "prompts.jsonl",
        out_dir / "trace_lengths.csv",
        out_dir / "trace_intervals.csv",
        out_dir / "workload_meta.json",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise SystemExit(f"Missing outputs: {missing}")

    trace_lengths = pd.read_csv(out_dir / "trace_lengths.csv")
    for c in ["request_id", "prompt_id", "num_prefill_tokens", "num_decode_tokens"]:
        if c not in trace_lengths.columns:
            raise SystemExit(f"trace_lengths.csv missing required column: {c}")

    trace_intervals = pd.read_csv(out_dir / "trace_intervals.csv")
    for c in ["request_id", "inter_arrival_ns", "arrival_time_ns"]:
        if c not in trace_intervals.columns:
            raise SystemExit(f"trace_intervals.csv missing required column: {c}")

    print(f"OK: {out_dir}")


if __name__ == "__main__":
    main()

