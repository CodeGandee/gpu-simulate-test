from __future__ import annotations

import argparse
import datetime
import gc
from pathlib import Path
from typing import Any


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
    import ray
    from vidur.profiling.cpu_overhead.benchmark_runner import BenchmarkRunner
    from vidur.profiling.utils import hex_to_binary

    if not ray.is_initialized():
        ray.init(include_dashboard=False, ignore_reinit_error=True)

    placement_group_ids = list(ray.util.placement_group_table().keys())
    for placement_group_id in placement_group_ids:
        ray._private.worker.global_worker.core_worker.remove_placement_group(
            ray.PlacementGroupID(hex_to_binary(placement_group_id))
        )

    # Avoid wrapping BenchmarkRunner itself in a Ray actor. On some hosts, Ray worker env handling
    # can expose unusable GPUs and cause PyTorch CUDA init to fail before Sarathi starts its own
    # GPU workers. Running BenchmarkRunner in-process keeps our `CUDA_VISIBLE_DEVICES` pinning
    # behavior deterministic.
    return BenchmarkRunner(model_name, batch_size, tensor_parallel_degree, output_dir, model_path)


def profile_model(
    model_name: str,
    batch_sizes_to_profile: list[int],
    tensor_parallel_degrees: list[int],
    output_dir: str,
    pbar: Any,
    logger: Any,
    model_path: str | None = None,
) -> None:
    import pandas as pd

    results: list[dict[str, Any]] = []
    last_error: str | None = None

    for tensor_parallel_degree in tensor_parallel_degrees:
        for batch_index, batch_size in enumerate(batch_sizes_to_profile):
            try:
                runner = _create_runner(
                    model_name, batch_size, tensor_parallel_degree, output_dir, model_path
                )
                results.append(runner.run())
                del runner
                gc.collect()
            except Exception as e:
                last_error = str(e)
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

    if not results:
        raise RuntimeError(
            "CPU overhead profiling produced 0 rows. This usually means the Sarathi/Ray workers failed to start "
            "or CUDA initialization failed on at least one visible GPU.\n"
            "\n"
            "Recommended next steps:\n"
            "- Pin to a known-good subset via `GSIM_CUDA_VISIBLE_DEVICES` (e.g. `0`) and rerun.\n"
            "- Inspect Ray logs under `/tmp/ray/session_latest/logs/` (or the newest `/tmp/ray/session_*/logs/`).\n"
            f"\nLast error: {last_error!r}\n"
        )

    df = pd.DataFrame(results)
    model_out_dir = Path(output_dir) / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(model_out_dir / "cpu_overhead.csv", index=False)


def main() -> None:
    from gpu_simulate_test.env_guard import (
        apply_cuda_visible_devices_from_gsim,
        patch_sarathi_preserve_cuda_visible_devices,
    )

    apply_cuda_visible_devices_from_gsim()
    patch_sarathi_preserve_cuda_visible_devices()

    from tqdm import tqdm
    from vidur.logger import init_logger
    from vidur.profiling.utils import get_cpu_overhead_batch_sizes_to_profile

    args = parse_args()
    logger = init_logger(__name__)

    batch_sizes_to_profile = get_cpu_overhead_batch_sizes_to_profile(args.max_batch_size)
    total = len(args.models) * len(args.num_tensor_parallel_workers) * len(batch_sizes_to_profile)
    pbar = tqdm(total=total)

    for model_name in args.models:
        try:
            profile_model(
                model_name=model_name,
                batch_sizes_to_profile=batch_sizes_to_profile,
                tensor_parallel_degrees=args.num_tensor_parallel_workers,
                output_dir=args.output_dir,
                pbar=pbar,
                logger=logger,
                model_path=args.model_path,
            )
        except Exception as e:
            logger.error("CPU overhead profiling failed for %s: %s", model_name, e)
            raise SystemExit(1) from e


if __name__ == "__main__":
    main()
