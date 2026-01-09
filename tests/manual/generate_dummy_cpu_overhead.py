from __future__ import annotations

from pathlib import Path

import pandas as pd

"""
Utility to generate a dummy Vidur-style CPU overhead CSV for local testing.

This is a manual script (not part of automated tests).
"""


def generate_dummy_cpu_overhead(
    output_path: Path,
    *,
    model_name: str = "meta-llama/Llama-2-7b-hf",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = []
    for bs in range(1, 129):
        row = {
            "model_name": model_name,
            "batch_size": bs,
            "tensor_parallel_degree": 1,
            "ray_comm_time_mean": 0.5,
            "schedule_median": 0.1,
            "sampler_e2e_median": 0.1,
            "prepare_inputs_e2e_median": 0.1,
            "process_model_outputs_median": 0.1,
            "model_execution_e2e_median": 10.0,
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated dummy CPU overheads at {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: generate_dummy_cpu_overhead.py <output_csv_path>")
    generate_dummy_cpu_overhead(Path(sys.argv[1]))
