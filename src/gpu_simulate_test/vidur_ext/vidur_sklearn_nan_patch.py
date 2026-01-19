"""
Opt-in per-target NaN handling for Vidur's sklearn execution-time predictor.

Vidur trains many per-op sklearn models from a single `mlp.csv`-derived dataframe. scikit-learn
does not accept NaNs in the target array `y`, which means a single missing timing target for an
op can cause training to fail.

This module provides a local, opt-in monkey patch that drops NaN rows *per target column* during
training:

- For each `_train_model(model_name, df, feature_cols, target_col)` call, we filter the training
  dataframe with `df.dropna(subset=[*feature_cols, target_col])`.
- Other ops keep their own available rows; we do not require a global "drop rows with any NaN"
  sanitization step.

This is intended for "best-effort" consumption of profiling roots when the effective NaN policy
is configured as `drop`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def patch_vidur_sklearn_train_model_dropna(*, summary: dict[str, Any]) -> Iterator[None]:
    """Patch Vidur's sklearn predictor to drop NaNs per target before training.

    Parameters
    ----------
    summary
        Mutable dict populated with per-model drop statistics during the patched region.
        The structure is JSON-serializable and safe to embed into run metadata.
    """
    try:
        from vidur.execution_time_predictor.sklearn_execution_time_predictor import (
            SklearnExecutionTimePredictor,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Failed to import Vidur's SklearnExecutionTimePredictor for patching.") from e

    if getattr(SklearnExecutionTimePredictor, "__gpu_simulate_test_dropna_patch__", None) is not None:
        raise RuntimeError("Vidur sklearn dropna patch is already active (nested patch contexts are not supported).")

    original_train_model = SklearnExecutionTimePredictor._train_model

    def _patched_train_model(self: Any, model_name: str, df: Any, feature_cols: Any, target_col: str) -> Any:
        total_rows = int(len(df))
        subset = [*list(feature_cols), str(target_col)]
        filtered = df.dropna(subset=subset)
        used_rows = int(len(filtered))
        dropped_rows = int(total_rows - used_rows)

        per_model = summary.setdefault("per_model", {})
        if isinstance(per_model, dict):
            per_model[str(model_name)] = {
                "model_name": str(model_name),
                "target_col": str(target_col),
                "feature_cols": [str(c) for c in list(feature_cols)],
                "rows_total": total_rows,
                "rows_used": used_rows,
                "rows_dropped": dropped_rows,
            }

        if used_rows <= 0:
            raise RuntimeError(
                "Vidur sklearn training has no usable rows after dropping NaNs for the current target. "
                f"model_name={model_name} target_col={target_col} rows_total={total_rows} rows_dropped={dropped_rows}. "
                "Remediation: rerun profiling with `profiling.mlp.profile_method=cuda_event` (or enable fallback) "
                "to produce complete timings."
            )

        return original_train_model(
            self,
            model_name=model_name,
            df=filtered,
            feature_cols=feature_cols,
            target_col=target_col,
        )

    setattr(SklearnExecutionTimePredictor, "__gpu_simulate_test_dropna_patch__", True)
    SklearnExecutionTimePredictor._train_model = _patched_train_model  # type: ignore[assignment]
    try:
        yield
    finally:
        SklearnExecutionTimePredictor._train_model = original_train_model  # type: ignore[assignment]
        try:
            delattr(SklearnExecutionTimePredictor, "__gpu_simulate_test_dropna_patch__")
        except AttributeError:  # pragma: no cover
            pass

