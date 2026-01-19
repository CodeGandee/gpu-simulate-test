"""
Profiling-root layout and validation utilities.

This module validates the existence and basic content of a Vidur-compatible profiling root
directory before it is consumed by simulation or reporting pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gpu_simulate_test.vidur_ext.mlp_validation import MlpValidationResult, validate_mlp_csv


@dataclass(frozen=True)
class ProfilingRootLayout:
    profiling_root: Path
    device: str
    model_id: str
    network_device: str = "a100_pairwise_nvlink"
    tensor_parallel_size: int = 1
    num_pipeline_stages: int = 1
    mlp_validation_mode: str = "strict"
    mlp_nan_policy: str = "auto"
    mlp_small_input_threshold: int = 128
    mlp_zero_heavy_limit: float = 0.01
    # Vidur's config uses the negative form: `skip_cpu_overhead_modeling`.
    skip_cpu_overhead_modeling: bool = True
    # Guardrails for fidelity runs. `strict` rejects placeholder-like CPU overhead CSVs.
    cpu_overhead_validation: str = "strict"


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


def validate_profiling_root(layout: ProfilingRootLayout) -> MlpValidationResult:
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

    mlp_csv = _compute_dir(layout) / "mlp.csv"
    mlp_result = validate_mlp_csv(
        mlp_csv,
        mode=str(layout.mlp_validation_mode).lower().strip() or "strict",  # type: ignore[arg-type]
        nan_policy=str(layout.mlp_nan_policy).lower().strip() or "auto",  # type: ignore[arg-type]
        small_input_threshold=int(layout.mlp_small_input_threshold),
        zero_heavy_limit=float(layout.mlp_zero_heavy_limit),
    )
    if mlp_result.warnings:
        import warnings

        for warning in mlp_result.warnings:
            warnings.warn(warning)

    if not layout.skip_cpu_overhead_modeling:
        from gpu_simulate_test.vidur_ext.cpu_overhead_validation import validate_cpu_overheads_csv

        csv_path = _cpu_overhead_dir(layout) / "cpu_overheads.csv"
        # This will raise in strict mode for empty/invalid/placeholder-like CSVs. In warn/off modes
        # it still raises for empty/invalid CSVs, but may allow placeholder-like inputs.
        cpu_result = validate_cpu_overheads_csv(
            csv_path,
            mode=str(layout.cpu_overhead_validation).lower().strip() or "strict",  # type: ignore[arg-type]
            expected_model_id=str(layout.model_id),
            expected_tensor_parallel_degree=int(layout.tensor_parallel_size),
        )
        if cpu_result.warnings:
            import warnings

            for warning in cpu_result.warnings:
                warnings.warn(warning)

    return mlp_result
