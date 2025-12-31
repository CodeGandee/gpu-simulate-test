from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class PathsConfig:
    """Repository-root anchored paths (Hydra-friendly)."""

    repo_root: Path
    tmp_root: Path


@dataclass(frozen=True)
class ModelConfig:
    """Model selection and tokenizer reference used across stages."""

    model_id: str
    tokenizer_ref: Path


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware identity used for provenance and profiling selection."""

    hardware_id: str
    device: str  # e.g., "cuda:0"


@dataclass(frozen=True)
class ArrivalScheduleConfig:
    """Deterministic arrival schedule parameters."""

    kind: Literal["fixed_interval", "poisson"]
    seed: int
    inter_arrival_ns: int = 0
    poisson_rate_per_s: float = 0.0


@dataclass(frozen=True)
class WorkloadConfig:
    """Workload-spec inputs shared across real+sim stages."""

    prompts: Path
    seed: int
    num_decode_tokens: int
    arrival: ArrivalScheduleConfig
    workload_dir: Path | None = None


@dataclass(frozen=True)
class ProfilingConfig:
    """Vidur profiling bundle location."""

    root: Path


@dataclass(frozen=True)
class BackendConfig:
    """Real benchmark backend selection."""

    name: Literal["sarathi", "transformers"]


@dataclass(frozen=True)
class OutputConfig:
    """Controls output directories and metadata writing."""

    run_id: str


@dataclass(frozen=True)
class StageConfig:
    """Shared stage metadata captured in run_meta/workload_meta."""

    run_type: Literal["workload", "real", "vidur", "compare", "vidur_profile"]


@dataclass(frozen=True)
class WorkloadPaths:
    """Absolute paths to workload artifacts."""

    workload_dir: Path
    prompts_jsonl: Path
    trace_lengths_csv: Path
    trace_intervals_csv: Path
    workload_meta_json: Path


@dataclass(frozen=True)
class RunPaths:
    """Absolute paths to standardized run artifacts."""

    run_dir: Path
    request_metrics_csv: Path
    token_metrics_csv: Path
    run_meta_json: Path


@dataclass(frozen=True)
class ReportPaths:
    """Absolute paths to comparison report artifacts."""

    out_dir: Path
    summary_md: Path
    tables_dir: Path
    figs_dir: Path
    run_meta_json: Path

