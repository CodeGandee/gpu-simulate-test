from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual smoke: real-bench (A100 required)")
    parser.add_argument("--backend", choices=["transformers", "sarathi"], default="transformers")
    parser.add_argument("--workload-dir", type=Path, required=True)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "gpu_simulate_test.cli.real_bench",
        f"backend={args.backend}",
        f"workload.workload_dir={args.workload_dir}",
    ]
    out = subprocess.check_output(cmd, cwd=_repo_root()).decode("utf-8").strip().splitlines()
    run_dir = Path(out[-1]).resolve()

    req_csv = run_dir / "request_metrics.csv"
    tok_csv = run_dir / "token_metrics.csv"
    if not req_csv.exists() or not tok_csv.exists():
        raise SystemExit(f"Missing outputs in {run_dir}")

    request_df = pd.read_csv(req_csv)
    for c in [
        "request_id",
        "arrival_time_ns",
        "first_token_time_ns",
        "ttft_ns",
        "num_prefill_tokens",
        "num_decode_tokens",
        "num_decode_tokens_actual",
        "status",
    ]:
        if c not in request_df.columns:
            raise SystemExit(f"request_metrics.csv missing required column: {c}")

    token_df = pd.read_csv(tok_csv)
    for c in ["request_id", "token_index", "token_time_ns", "token_latency_ns"]:
        if c not in token_df.columns:
            raise SystemExit(f"token_metrics.csv missing required column: {c}")

    # Monotonic token_time_ns within each request_id.
    for rid, g in token_df.sort_values(["request_id", "token_index"]).groupby("request_id"):
        times = g["token_time_ns"].astype(int).tolist()
        if times != sorted(times):
            raise SystemExit(f"Non-monotonic token_time_ns for request_id={rid}")

    print(f"OK: {run_dir}")


if __name__ == "__main__":
    main()

