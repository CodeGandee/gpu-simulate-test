from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gpu_simulate_test.io import stable_id, write_json
from gpu_simulate_test.paper_fidelity.capacity import CapacityCriterion, discover_capacity, write_capacity_json
from gpu_simulate_test.paper_fidelity.paths import PaperFidelityPaths
from gpu_simulate_test.paper_fidelity.traces import TraceSpec, add_poisson_arrivals, processed_lengths_csv_to_trace
from gpu_simulate_test.real_bench.backends.sarathi_paper_fidelity_backend import (
    SarathiPaperFidelityInputs,
    run_sarathi_paper_fidelity,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual smoke: paper-fidelity capacity discovery (CUDA required)")
    parser.add_argument("--scenario", default="llama2_7b_arxiv_smoke")
    parser.add_argument("--model-id", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--model-ref", type=Path, default=_repo_root() / "models" / "llama2-7b-hf" / "source-data")
    parser.add_argument("--min-qps", type=float, default=1.0)
    parser.add_argument("--max-qps", type=float, default=2.0)
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--num-requests", type=int, default=8)
    args = parser.parse_args()

    repo_root = _repo_root()
    pf_paths = PaperFidelityPaths(repo_root=repo_root)

    base = processed_lengths_csv_to_trace(
        repo_root
        / "extern"
        / "tracked"
        / "vidur"
        / "data"
        / "processed_traces"
        / "arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv",
        spec=TraceSpec(max_tokens=4096, seed=42, num_requests=int(args.num_requests)),
    )

    capacity_dir = pf_paths.capacity_dir(args.scenario)
    capacity_dir.mkdir(parents=True, exist_ok=True)

    run_id = stable_id([args.scenario, str(args.min_qps), str(args.max_qps)], prefix="pf_capacity", length=12)
    criterion = CapacityCriterion(metric="request_scheduling_delay", quantile=0.99, threshold_s=5.0)

    def run_at_qps(qps: float) -> pd.DataFrame:
        trace_df = add_poisson_arrivals(base, qps=float(qps), seed=42)
        trace_csv = capacity_dir / "trace.csv"
        trace_df.to_csv(trace_csv, index=False)
        out_dir = capacity_dir / f"qps_{qps:.4f}"
        req_csv = run_sarathi_paper_fidelity(
            SarathiPaperFidelityInputs(
                scenario_name=args.scenario,
                trace_csv=trace_csv,
                model_id=args.model_id,
                model_ref=args.model_ref,
                seed=42,
                max_tokens=4096,
            ),
            out_dir=out_dir,
        )
        return pd.read_csv(req_csv)

    result = discover_capacity(
        run_at_qps=run_at_qps,
        min_qps=float(args.min_qps),
        max_qps=float(args.max_qps),
        max_iters=int(args.max_iters),
        criterion=criterion,
        operating_point_fraction=0.85,
    )

    write_capacity_json(capacity_dir / "capacity.json", result=result)
    write_json(capacity_dir / "run_meta.json", {"schema_version": "v1", "run_type": "capacity", "run_id": run_id})

    print(f"OK: {capacity_dir}")


if __name__ == "__main__":
    main()

