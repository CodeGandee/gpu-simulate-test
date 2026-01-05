from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_trace(*, scenario: str, workload: str, overrides: list[str]) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "gpu_simulate_test.cli.paper_fidelity",
        "trace",
        "--scenario",
        scenario,
        "--workload",
        workload,
        *overrides,
    ]
    out = subprocess.check_output(cmd, cwd=_repo_root()).decode("utf-8").strip().splitlines()
    return Path(out[-1]).resolve()


def main() -> None:
    static_dir = _run_trace(
        scenario="llama2_7b_arxiv",
        workload="static",
        overrides=[
            "scenario.name=llama2_7b_arxiv_smoke_static",
            "scenario.trace_source.num_requests=8",
        ],
    )
    static_csv = static_dir / "trace.csv"
    if not static_csv.exists():
        raise SystemExit(f"Missing trace.csv at {static_csv}")
    static_df = pd.read_csv(static_csv)
    if not (static_df["arrived_at"].astype(float) == 0.0).all():
        raise SystemExit("Static trace arrived_at must be all zeros")

    dynamic_dir = _run_trace(
        scenario="llama2_7b_arxiv",
        workload="dynamic",
        overrides=[
            "scenario.name=llama2_7b_arxiv_smoke_dynamic",
            "scenario.trace_source.num_requests=8",
            "workload.qps=2.0",
            "workload.seed=123",
        ],
    )
    dynamic_csv = dynamic_dir / "trace.csv"
    if not dynamic_csv.exists():
        raise SystemExit(f"Missing trace.csv at {dynamic_csv}")
    first = dynamic_csv.read_bytes()

    dynamic_dir_2 = _run_trace(
        scenario="llama2_7b_arxiv",
        workload="dynamic",
        overrides=[
            "scenario.name=llama2_7b_arxiv_smoke_dynamic",
            "scenario.trace_source.num_requests=8",
            "workload.qps=2.0",
            "workload.seed=123",
        ],
    )
    dynamic_csv_2 = dynamic_dir_2 / "trace.csv"
    second = dynamic_csv_2.read_bytes()
    if first != second:
        raise SystemExit("Dynamic trace is not deterministic for a fixed seed")

    print(f"OK: {static_dir} {dynamic_dir}")


if __name__ == "__main__":
    main()

