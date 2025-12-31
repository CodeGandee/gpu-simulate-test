from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProfilingRootLayout:
    profiling_root: Path
    device: str
    model_id: str
    network_device: str = "a100_pairwise_nvlink"
    tensor_parallel_size: int = 1
    num_pipeline_stages: int = 1
    skip_cpu_overhead_modeling: bool = True


def _compute_dir(layout: ProfilingRootLayout) -> Path:
    return layout.profiling_root / "data" / "profiling" / "compute" / layout.device / layout.model_id


def _network_dir(layout: ProfilingRootLayout) -> Path:
    return layout.profiling_root / "data" / "profiling" / "network" / layout.network_device


def _cpu_overhead_dir(layout: ProfilingRootLayout) -> Path:
    return (
        layout.profiling_root
        / "data"
        / "profiling"
        / "cpu_overhead"
        / layout.network_device
        / layout.model_id
    )


def validate_profiling_root(layout: ProfilingRootLayout) -> None:
    if not layout.profiling_root.exists():
        raise FileNotFoundError(f"profiling_root does not exist: {layout.profiling_root}")

    required: list[Path] = [
        _compute_dir(layout) / "mlp.csv",
        _compute_dir(layout) / "attention.csv",
    ]

    if layout.num_pipeline_stages > 1:
        required.append(_network_dir(layout) / "send_recv.csv")
    if layout.tensor_parallel_size > 1:
        required.append(_network_dir(layout) / "all_reduce.csv")
    if not layout.skip_cpu_overhead_modeling:
        required.append(_cpu_overhead_dir(layout) / "cpu_overheads.csv")

    missing = [p for p in required if not p.exists()]
    if missing:
        msg = "Missing profiling inputs:\n" + "\n".join([f"- {p}" for p in missing])
        raise FileNotFoundError(msg)

