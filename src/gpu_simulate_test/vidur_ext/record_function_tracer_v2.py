"""
Record-function tracer extension for Vidur MLP profiling.

Vidur's upstream record-function tracer only attributes GPU time via `cuda_runtime` launch events.
On some hosts/models, kernels can be launched via the CUDA driver path (`cuda_driver`, e.g.
`cuLaunchKernel`). When this happens, upstream attribution can miss GPU execution time and produce
missing per-op samples.

This module provides `RecordFunctionTracerV2`, a drop-in replacement that:

- Collects correlation ids from both `cuda_runtime` and `cuda_driver` launch events within each
  `user_annotation` region.
- Attributes GPU execution time from correlated `kernel` events back to each annotated region.
- Produces per-op summary statistics in milliseconds matching Vidur's expected shape.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _find_children(trace_events: list[dict[str, Any]], parent: dict[str, Any]) -> list[dict[str, Any]]:
    if "ts" not in parent or "dur" not in parent:
        return []

    parent_start = parent["ts"]
    parent_end = parent["ts"] + parent["dur"]

    children: list[dict[str, Any]] = []
    for event in trace_events:
        if "ts" not in event or "dur" not in event:
            continue
        start = event["ts"]
        end = event["ts"] + event["dur"]
        if start > parent_start and end < parent_end:
            children.append(event)
    return children


def compute_operation_time_stats(trace_events: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute per-op summary stats in milliseconds from chrome trace events.

    Parameters
    ----------
    trace_events
        List of `traceEvents` entries from a torch profiler chrome trace.

    Returns
    -------
    dict
        Mapping from operation name (e.g., `mlp_up_proj`) to a dict containing `min`, `max`,
        `mean`, `median`, and `std` in milliseconds.
    """
    launch_categories = {"cuda_runtime", "cuda_driver"}
    execution_categories = {"kernel", "gpu_memcpy", "gpu_memset"}

    correlated_gpu_us: dict[object, float] = defaultdict(float)
    for event in trace_events:
        if event.get("cat") not in execution_categories:
            continue
        args = event.get("args") or {}
        if "correlation" not in args or "dur" not in event:
            continue
        correlated_gpu_us[args["correlation"]] += float(event["dur"])

    samples_ms_by_op: dict[str, list[float]] = defaultdict(list)
    for region in trace_events:
        if region.get("cat") != "user_annotation":
            continue
        region_name = str(region.get("name") or "")
        if not region_name:
            continue

        corr_ids: set[object] = set()
        for child in _find_children(trace_events, region):
            if child.get("cat") not in launch_categories:
                continue
            args = child.get("args") or {}
            corr = args.get("correlation")
            if corr is not None:
                corr_ids.add(corr)

        gpu_us = sum(correlated_gpu_us.get(corr, 0.0) for corr in corr_ids)
        if gpu_us <= 0.0:
            continue

        op_name = region_name.removeprefix("vidur_")
        samples_ms_by_op[op_name].append(gpu_us * 1e-3)

    summary: dict[str, dict[str, float]] = {}
    for op_name, samples in samples_ms_by_op.items():
        arr = np.asarray(samples, dtype=float)
        summary[op_name] = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
        }
    return summary


@dataclass
class RecordFunctionTracerV2:
    """Correlation-based op timing tracer for Vidur MLP profiling.

    This mirrors Vidur's `RecordFunctionTracer` interface so it can be monkey-patched into
    `vidur.profiling.mlp.mlp_wrapper` at runtime.
    """

    output_path: str
    trace_path: Path = field(init=False)
    profiler: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        trace_id = str(uuid.uuid4())[:8]
        self.trace_path = (
            Path(str(self.output_path)) / "profiler_traces" / f"profiler_trace_{trace_id}.json"
        )

    def __enter__(self) -> None:  # pragma: no cover
        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "torch is required for record-function profiling; run inside the Pixi env (`pixi install`)."
            ) from e

        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        self.profiler.__enter__()

    def __exit__(self, *args: object) -> None:  # pragma: no cover
        if self.profiler is None:
            return

        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "torch is required for record-function profiling; run inside the Pixi env (`pixi install`)."
            ) from e

        self.profiler.__exit__(None, None, None)
        torch.cuda.synchronize()

        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.profiler.export_chrome_trace(str(self.trace_path))

    def get_operation_time_stats(self) -> dict[str, dict[str, float]]:
        """Load the exported trace and return per-op summary statistics (milliseconds)."""
        payload = json.loads(self.trace_path.read_text(encoding="utf-8"))
        trace_events = payload.get("traceEvents") or []
        if not isinstance(trace_events, list):
            raise ValueError(f"Unexpected traceEvents payload in {self.trace_path}")
        return compute_operation_time_stats(trace_events)

