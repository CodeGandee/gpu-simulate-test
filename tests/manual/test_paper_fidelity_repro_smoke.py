from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual smoke: paper-fidelity repro (Vidur + CUDA required)")
    parser.add_argument("--scenario", default="llama2_7b_arxiv")
    parser.add_argument("--workload", choices=["static", "dynamic"], default="static")
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    scenario_name = f"{args.scenario}_smoke_repro"

    cmd = [
        sys.executable,
        "-m",
        "gpu_simulate_test.cli.paper_fidelity",
        "repro",
        "--scenario",
        args.scenario,
        "--workload",
        args.workload,
        f"scenario.name={scenario_name}",
        f"scenario.trace_source.num_requests={int(args.num_requests)}",
        f"scenario.trace_source.max_tokens={int(args.max_tokens)}",
    ]
    out = subprocess.check_output(cmd, cwd=_repo_root()).decode("utf-8").strip().splitlines()
    report_line = next((line for line in reversed(out) if line.startswith("/") and "/results/reports/" in line), None)
    if report_line is None:
        raise SystemExit("Could not find report directory in command output:\n" + "\n".join(out[-20:]))
    report_dir = Path(report_line).resolve()

    repo_root = _repo_root()
    trace_csv = repo_root / "tmp" / "paper_fidelity" / "traces" / scenario_name / "trace.csv"
    sim_csv = repo_root / "tmp" / "paper_fidelity" / "runs" / scenario_name / "sim" / "request_metrics.csv"
    real_csv = repo_root / "tmp" / "paper_fidelity" / "runs" / scenario_name / "real" / "request_metrics.csv"
    summary_md = report_dir / "summary.md"

    missing = [p for p in [trace_csv, sim_csv, real_csv, summary_md] if not p.exists()]
    if missing:
        raise SystemExit("Missing expected outputs:\n" + "\n".join([f"- {p}" for p in missing]))

    print(f"OK: {report_dir}")


if __name__ == "__main__":
    main()
