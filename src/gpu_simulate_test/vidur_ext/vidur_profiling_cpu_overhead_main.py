from __future__ import annotations

import argparse
import datetime
import gc
import os
from pathlib import Path
from typing import Any

import pandas as pd
import ray
from tqdm import tqdm

from vidur.logger import init_logger
from vidur.profiling.cpu_overhead.benchmark_runner import BenchmarkRunner
from vidur.profiling.utils import get_cpu_overhead_batch_sizes_to_profile, hex_to_binary

logger = init_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU Overhead Profiling (patched wrapper)")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Staging root for profiling results (writes under <output_dir>/cpu_overhead/<timestamp>/...)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "internlm/internlm-20b",
            "Qwen/Qwen-72B",
            "meta-llama/Llama-2-7b-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "meta-llama/Llama-2-70b-hf",
        ],
        help="Models to profile",
    )
    parser.add_argument(
        "--num_tensor_parallel_workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Number of tensor parallel workers to profile",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=128,
        help="Maximum batch size to profile",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Local path to the model (optional)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir).expanduser()
    run_dir = output_root / "cpu_overhead" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(run_dir)

    return args


def _create_runner(
    model_name: str,
    batch_size: int,
    tensor_parallel_degree: int,
    output_dir: str,
    model_path: str | None = None,
) -> Any:
    if not ray.is_initialized():
        ray.init(include_dashboard=False, ignore_reinit_error=True)

    placement_group_ids = list(ray.util.placement_group_table().keys())
    for placement_group_id in placement_group_ids:
        ray._private.worker.global_worker.core_worker.remove_placement_group(
            ray.PlacementGroupID(hex_to_binary(placement_group_id))
        )

    runner_class = (
        ray.remote(num_gpus=0)(BenchmarkRunner)
        .options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})
        .remote
    )

    return runner_class(model_name, batch_size, tensor_parallel_degree, output_dir, model_path)


def profile_model(
    model_name: str,
    batch_sizes_to_profile: list[int],
    tensor_parallel_degrees: list[int],
    output_dir: str,
    pbar: Any,
    model_path: str | None = None,
) -> None:
    results: list[dict[str, Any]] = []

    for tensor_parallel_degree in tensor_parallel_degrees:
        for batch_index, batch_size in enumerate(batch_sizes_to_profile):
            try:
                runner = _create_runner(
                    model_name, batch_size, tensor_parallel_degree, output_dir, model_path
                )
                results.append(ray.get(runner.run.remote()))
                del runner
                gc.collect()
            except Exception as e:
                logger.error(
                    "Failed to run %s_%s_%s due to %s",
                    model_name,
                    batch_size,
                    tensor_parallel_degree,
                    e,
                )
                pbar.update(len(batch_sizes_to_profile) - batch_index)
                break

            pbar.update(1)

    df = pd.DataFrame(results)
    model_out_dir = Path(output_dir) / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(model_out_dir / "cpu_overhead.csv", index=False)


def main() -> None:
    args = parse_args()

    batch_sizes_to_profile = get_cpu_overhead_batch_sizes_to_profile(args.max_batch_size)
    total = len(args.models) * len(args.num_tensor_parallel_workers) * len(batch_sizes_to_profile)
    pbar = tqdm(total=total)

    for model_name in args.models:
        profile_model(
            model_name=model_name,
            batch_sizes_to_profile=batch_sizes_to_profile,
            tensor_parallel_degrees=args.num_tensor_parallel_workers,
            output_dir=args.output_dir,
            pbar=pbar,
            model_path=args.model_path,
        )


if __name__ == "__main__":
    main()
