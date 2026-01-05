from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gpu_simulate_test.io import stable_id, write_json
from gpu_simulate_test.paper_fidelity.paths import PaperFidelityPaths
from gpu_simulate_test.real_bench.backends.sarathi_paper_fidelity_backend import (
    SarathiPaperFidelityInputs,
    run_sarathi_paper_fidelity,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual smoke: paper-fidelity Sarathi real replay (CUDA required)")
    parser.add_argument("--scenario", default="llama2_7b_arxiv_smoke")
    parser.add_argument("--model-id", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--model-ref", type=Path, default=_repo_root() / "models" / "llama2-7b-hf" / "source-data")
    args = parser.parse_args()

    repo_root = _repo_root()
    pf_paths = PaperFidelityPaths(repo_root=repo_root)

    trace_dir = pf_paths.trace_dir(args.scenario)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_csv = trace_dir / "trace.csv"

    trace_df = pd.DataFrame(
        {
            "arrived_at": [0.0, 0.0, 0.0],
            "num_prefill_tokens": [8, 8, 8],
            "num_decode_tokens": [4, 4, 4],
            "request_id": [0, 1, 2],
        }
    )
    trace_df.to_csv(trace_csv, index=False)
    write_json(trace_dir / "trace_meta.json", {"schema_version": "v1", "note": "manual smoke trace"})

    run_id = stable_id([args.scenario, str(trace_csv)], prefix="pf_real", length=12)
    real_dir = pf_paths.real_dir(args.scenario)

    run_sarathi_paper_fidelity(
        SarathiPaperFidelityInputs(
            scenario_name=args.scenario,
            trace_csv=trace_csv,
            model_id=args.model_id,
            model_ref=args.model_ref,
            seed=int(run_id.split("_")[-1], 16) % 2**31,
            max_tokens=4096,
        ),
        out_dir=real_dir,
    )

    req_csv = real_dir / "request_metrics.csv"
    meta_json = real_dir / "run_meta.json"
    if not (req_csv.exists() and meta_json.exists()):
        raise SystemExit(f"Missing outputs under {real_dir}")

    out_df = pd.read_csv(req_csv)
    for c in [
        "request_id",
        "request_scheduling_delay",
        "request_execution_plus_preemption_time_normalized",
        "request_e2e_time_normalized",
    ]:
        if c not in out_df.columns:
            raise SystemExit(f"request_metrics.csv missing required column: {c}")

    print(f"OK: {real_dir}")


if __name__ == "__main__":
    main()

