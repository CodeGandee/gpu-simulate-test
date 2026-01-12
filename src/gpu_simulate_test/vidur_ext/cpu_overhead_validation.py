from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


CpuOverheadValidationMode = Literal["strict", "warn", "off"]


@dataclass(frozen=True)
class CpuOverheadsValidationResult:
    csv_path: Path
    mode: CpuOverheadValidationMode
    row_count: int
    column_count: int
    placeholder_like: bool
    batch_size_unique: int
    unique_values: dict[str, int]
    warnings: list[str]

    def as_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        data["csv_path"] = str(self.csv_path)
        return data


_REQUIRED_COLUMNS: tuple[str, ...] = ("model_name", "batch_size", "tensor_parallel_degree")

_PLACEHOLDER_CHECK_COLUMNS: tuple[str, ...] = (
    # Core output from Vidur's CPU overhead benchmark runner.
    "ray_comm_time_mean",
    "schedule_median",
    "sampler_e2e_median",
    "prepare_inputs_e2e_median",
    "process_model_outputs_median",
    "model_execution_e2e_median",
    # Keep means as fallback in case medians are missing in a future schema.
    "schedule_mean",
    "sampler_e2e_mean",
    "prepare_inputs_e2e_mean",
    "process_model_outputs_mean",
    "model_execution_e2e_mean",
)


def validate_cpu_overheads_csv(
    csv_path: Path,
    *,
    mode: CpuOverheadValidationMode = "strict",
    expected_model_id: str | None = None,
    expected_tensor_parallel_degree: int | None = None,
) -> CpuOverheadsValidationResult:
    """Validate a Vidur-style `cpu_overheads.csv` (or `cpu_overhead.csv`) file.

    This validator is intentionally lightweight and defensive: it aims to catch the most common
    failure modes in paper-fidelity workflows:
    - the profiler produced an empty (or headerless) CSV
    - the CSV is missing expected identifier columns
    - the CSV looks like placeholder/dummy data (all key overhead columns constant)
    """
    if mode not in {"strict", "warn", "off"}:
        raise ValueError(f"Unsupported cpu_overhead validation mode: {mode!r} (expected strict|warn|off)")

    if not csv_path.exists():
        raise FileNotFoundError(f"CPU overhead CSV does not exist: {csv_path}")

    import pandas as pd

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as e:  # pragma: no cover
        raise ValueError(
            "CPU overhead CSV is empty (no header/rows). This usually means CPU overhead profiling failed. "
            f"Path: {csv_path}"
        ) from e

    if df.empty:
        raise ValueError(
            "CPU overhead CSV has 0 rows. This usually means CPU overhead profiling failed. "
            f"Path: {csv_path}"
        )

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "CPU overhead CSV is missing required columns "
            f"{missing}. Path: {csv_path}. Columns: {sorted(df.columns)}"
        )

    warnings: list[str] = []

    if expected_model_id is not None:
        model_values = {str(v) for v in df["model_name"].dropna().unique().tolist()}
        if model_values and model_values != {expected_model_id}:
            warnings.append(
                "CPU overhead CSV model_name values do not match the expected model id: "
                f"expected={expected_model_id!r}, observed={sorted(model_values)!r}."
            )

    if expected_tensor_parallel_degree is not None:
        try:
            tp_values = {int(v) for v in df["tensor_parallel_degree"].dropna().unique().tolist()}
        except Exception:
            tp_values = set()
        if tp_values and tp_values != {int(expected_tensor_parallel_degree)}:
            warnings.append(
                "CPU overhead CSV tensor_parallel_degree values do not match the expected TP degree: "
                f"expected={int(expected_tensor_parallel_degree)!r}, observed={sorted(tp_values)!r}."
            )

    batch_size_unique = int(df["batch_size"].nunique(dropna=True))
    candidate_cols = [c for c in _PLACEHOLDER_CHECK_COLUMNS if c in df.columns]
    unique_values: dict[str, int] = {}
    for col in candidate_cols:
        try:
            unique_values[col] = int(df[col].nunique(dropna=False))
        except Exception:
            unique_values[col] = 0

    placeholder_like = False
    if candidate_cols and batch_size_unique >= 2 and len(df) >= 4:
        # "Dummy" data in this repo sets every key overhead column constant across the entire grid.
        placeholder_like = all(unique_values.get(col, 0) <= 1 for col in candidate_cols)

    if placeholder_like:
        msg = (
            "CPU overhead CSV looks like placeholder/dummy data: key overhead columns are constant across rows. "
            "This commonly happens when the profiler failed and a synthetic CSV was substituted "
            f"(see `tests/manual/generate_dummy_cpu_overhead.py`). Path: {csv_path}"
        )
        if mode == "strict":
            raise ValueError(msg)
        if mode == "warn":
            warnings.append(msg)

    return CpuOverheadsValidationResult(
        csv_path=csv_path,
        mode=mode,
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        placeholder_like=bool(placeholder_like),
        batch_size_unique=batch_size_unique,
        unique_values=unique_values,
        warnings=warnings,
    )

