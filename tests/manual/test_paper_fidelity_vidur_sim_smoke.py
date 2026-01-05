from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gpu_simulate_test.io import read_json, stable_id, write_json
from gpu_simulate_test.paper_fidelity.paths import PaperFidelityPaths, build_run_meta
from gpu_simulate_test.vidur_ext.sim_runner import VidurPaperFidelitySimInputs, run_vidur_paper_fidelity_sim


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual smoke: paper-fidelity Vidur sim (profiling required)")
    parser.add_argument("--scenario", default="llama2_7b_arxiv_smoke")
    parser.add_argument("--profiling-root", type=Path, default=_repo_root() / "extern" / "tracked" / "vidur")
    parser.add_argument("--model-id", default="meta-llama/Llama-2-7b-hf")
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
        }
    )
    trace_df.to_csv(trace_csv, index=False)
    write_json(trace_dir / "trace_meta.json", {"schema_version": "v1", "note": "manual smoke trace"})

    run_id = stable_id([args.scenario, str(trace_csv)], prefix="pf_vidur_sim", length=12)
    sim_dir = pf_paths.sim_dir(args.scenario)

    meta = build_run_meta(
        repo_root=repo_root,
        run_type="sim",
        run_id=run_id,
        scenario_name=args.scenario,
        artifacts={"trace_csv": trace_csv, "sim_dir": sim_dir},
    )

    run_vidur_paper_fidelity_sim(
        VidurPaperFidelitySimInputs(
            scenario_name=args.scenario,
            trace_csv=trace_csv,
            profiling_root=args.profiling_root,
            model_id=args.model_id,
        ),
        out_dir=sim_dir,
        run_meta=meta,
    )

    req_csv = sim_dir / "request_metrics.csv"
    meta_json = sim_dir / "run_meta.json"
    if not (req_csv.exists() and meta_json.exists()):
        raise SystemExit(f"Missing outputs under {sim_dir}")

    meta = read_json(meta_json)
    raw_dir = Path(meta["vidur_raw_dir"]).resolve()
    raw_req_csv = raw_dir / "request_metrics.csv"
    if not raw_req_csv.exists():
        raise SystemExit(f"Missing raw Vidur request_metrics.csv under {raw_dir}")

    out_df = pd.read_csv(req_csv)
    for c in [
        "request_id",
        "request_scheduling_delay",
        "request_execution_plus_preemption_time_normalized",
        "request_e2e_time_normalized",
    ]:
        if c not in out_df.columns:
            raise SystemExit(f"request_metrics.csv missing required column: {c}")

    print(f"OK: {sim_dir}")


if __name__ == "__main__":
    main()
