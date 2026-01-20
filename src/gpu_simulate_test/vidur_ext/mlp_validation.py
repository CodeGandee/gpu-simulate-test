"""
MLP profiling CSV validation helpers.

This module provides a lightweight validator for Vidur-style `mlp.csv` artifacts. It detects:
- Missing timing targets (any missing value in core `time_stats.*.{min,max,mean,median}` columns)
- Suspiciously zero-heavy timing targets above a configured token threshold

Missing-value handling is controlled by `nan_policy`:

- `reject` fails on any missing core timing target cell (default when mode=strict + nan_policy=auto)
- `drop` allows missing cells (consumers must drop missing samples per target before sklearn training)

Zero-heavy handling is controlled by `mode`:

- `strict` fails on zero-heavy signals
- `non_strict` warns on zero-heavy signals
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


MlpValidationMode = Literal["strict", "non_strict"]
MlpNanPolicy = Literal["auto", "reject", "drop", "zero"]
MlpEffectiveNanPolicy = Literal["reject", "drop", "zero"]


@dataclass(frozen=True)
class MlpValidationResult:
    csv_path: Path
    mode: MlpValidationMode
    nan_policy: MlpNanPolicy
    nan_policy_effective: MlpEffectiveNanPolicy
    row_count: int
    column_count: int
    time_column_count: int
    missing_cells_total: int
    missing_columns: list[str]
    zero_heavy_columns: list[str]
    thresholds: dict[str, Any]
    warnings: list[str]

    def as_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        data["csv_path"] = str(self.csv_path)
        return data


class MlpCsvValidationError(ValueError):
    def __init__(self, message: str, *, result: MlpValidationResult) -> None:
        super().__init__(message)
        self.result = result


_CORE_TIME_STATS: frozenset[str] = frozenset({"min", "max", "mean", "median"})


def resolve_nan_policy(*, mode: MlpValidationMode, nan_policy: MlpNanPolicy) -> MlpEffectiveNanPolicy:
    """Resolve the effective NaN policy based on mode and configuration."""
    if nan_policy not in {"auto", "reject", "drop", "zero"}:
        raise ValueError(f"Unsupported nan_policy: {nan_policy!r} (expected auto|reject|drop|zero)")
    if nan_policy == "auto":
        return "reject" if mode == "strict" else "drop"
    if nan_policy == "reject":
        return "reject"
    if nan_policy == "drop":
        return "drop"
    return "zero"


def validate_mlp_csv(
    csv_path: Path,
    *,
    mode: MlpValidationMode = "strict",
    nan_policy: MlpNanPolicy = "auto",
    small_input_threshold: int = 128,
    zero_heavy_limit: float = 0.01,
) -> MlpValidationResult:
    """Validate a staged Vidur-style MLP profiling CSV.

    Validation rules:
    - Missing columns always fail.
    - Missing cells (NaNs in core time_stats targets) are handled by `nan_policy`:
      - reject: fail if any core cell is missing
      - drop: allow missing cells and record warnings (consumers must handle NaNs)
    - Zero-heavy signals (many exact zeros above a token threshold) fail in strict mode and warn in
      non-strict mode.
    """
    if mode not in {"strict", "non_strict"}:
        raise ValueError(f"Unsupported MLP validation mode: {mode!r} (expected strict|non_strict)")

    effective_nan_policy = resolve_nan_policy(mode=mode, nan_policy=nan_policy)

    if zero_heavy_limit < 0.0 or zero_heavy_limit > 1.0:
        raise ValueError(f"zero_heavy_limit must be within [0, 1] (got {zero_heavy_limit!r})")

    if small_input_threshold < 0:
        raise ValueError(f"small_input_threshold must be >= 0 (got {small_input_threshold!r})")

    if not csv_path.exists():
        raise FileNotFoundError(f"MLP CSV does not exist: {csv_path}")

    import pandas as pd

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as e:  # pragma: no cover
        raise ValueError(f"MLP CSV is empty (no header/rows): {csv_path}") from e

    if df.empty:
        raise ValueError(f"MLP CSV has 0 rows: {csv_path}")

    time_cols: list[str] = []
    time_cols_by_op: dict[str, set[str]] = {}
    for col in df.columns:
        if not str(col).startswith("time_stats."):
            continue
        parts = str(col).split(".")
        if len(parts) < 3:
            continue
        stat = parts[-1]
        if stat not in _CORE_TIME_STATS:
            continue
        op_name = ".".join(parts[1:-1])
        if not op_name:
            continue
        time_cols.append(str(col))
        time_cols_by_op.setdefault(op_name, set()).add(stat)

    missing_columns: list[str] = []
    if "num_tokens" not in df.columns:
        missing_columns.append("num_tokens")

    for op_name, stats in sorted(time_cols_by_op.items()):
        missing_stats = sorted(_CORE_TIME_STATS.difference(stats))
        for stat in missing_stats:
            missing_columns.append(f"time_stats.{op_name}.{stat}")

    if not time_cols:
        missing_columns.append("time_stats.*.{min,max,mean,median}")

    missing_cells_total = 0
    if time_cols:
        missing_cells_total = int(df[time_cols].isna().sum().sum())

    warnings: list[str] = []
    zero_heavy_columns: list[str] = []
    thresholds = {
        "small_input_threshold": int(small_input_threshold),
        "zero_heavy_limit": float(zero_heavy_limit),
    }

    if missing_cells_total > 0 and effective_nan_policy == "drop":
        warnings.append(
            "Missing (NaN) timing targets detected in mlp.csv. nan_policy=drop allows this, but "
            "consumers must drop NaN samples per target before training. Consider rerunning with "
            "`profiling.mlp.profile_method=cuda_event` (or enable fallback) for highest fidelity."
        )
    if missing_cells_total > 0 and effective_nan_policy == "zero":
        warnings.append(
            "Missing (NaN) timing targets detected in mlp.csv. nan_policy=zero allows this, but "
            "consumers will fill NaN targets with 0 during training. This can bias predictors and "
            "worsen sim-vs-real fidelity. Consider rerunning with `profiling.mlp.profile_method=cuda_event` "
            "(or enable fallback) for highest fidelity."
        )

    if "num_tokens" in df.columns and time_cols:
        try:
            num_tokens = df["num_tokens"].astype(int)
        except Exception as e:
            raise ValueError(f"MLP CSV has non-integer num_tokens values: {csv_path}") from e

        eligible = num_tokens >= int(small_input_threshold)
        if bool(eligible.any()):
            for col in time_cols:
                series = df.loc[eligible, col]
                denom = int(series.notna().sum())
                if denom <= 0:
                    continue
                zeros = int((series == 0).sum())
                zero_rate = zeros / float(denom)
                if zero_rate > float(zero_heavy_limit):
                    zero_heavy_columns.append(col)
                    warnings.append(
                        f"Zero-heavy timing targets for num_tokens>={int(small_input_threshold)}: "
                        f"{col} has zero_rate={zero_rate:.4f} (limit={float(zero_heavy_limit):.4f})."
                    )

    result = MlpValidationResult(
        csv_path=csv_path,
        mode=mode,
        nan_policy=nan_policy,
        nan_policy_effective=effective_nan_policy,
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        time_column_count=int(len(time_cols)),
        missing_cells_total=int(missing_cells_total),
        missing_columns=missing_columns,
        zero_heavy_columns=zero_heavy_columns,
        thresholds=thresholds,
        warnings=warnings,
    )

    failed_missing = bool(missing_columns) or (int(missing_cells_total) > 0 and effective_nan_policy == "reject")
    failed_zero_heavy = bool(zero_heavy_columns) and mode == "strict"
    if failed_missing or failed_zero_heavy:
        problems: list[str] = []
        if missing_columns:
            problems.append(f"missing_columns={missing_columns}")
        if missing_cells_total > 0:
            problems.append(f"missing_cells_total={int(missing_cells_total)}")
        if zero_heavy_columns:
            problems.append(f"zero_heavy_columns={zero_heavy_columns}")

        remediation = (
            "Remediation: rerun with a different MLP profiling method, e.g. "
            "`profiling.mlp.profile_method=cuda_event`, or enable automatic fallback via "
            "`profiling.mlp.fallback.enabled=true profiling.mlp.fallback.method=cuda_event`."
        )
        raise MlpCsvValidationError(
            "MLP profiling CSV failed validation. "
            f"csv_path={csv_path} mode={mode} nan_policy={nan_policy} thresholds={thresholds} "
            + " ".join(problems)
            + f". {remediation}",
            result=result,
        )

    return result
