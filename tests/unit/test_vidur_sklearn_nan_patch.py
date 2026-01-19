from __future__ import annotations

import pandas as pd
import pytest

from gpu_simulate_test.vidur_ext.vidur_sklearn_nan_patch import patch_vidur_sklearn_train_model_dropna


def test_patch_vidur_sklearn_train_model_dropna_filters_and_records(monkeypatch: pytest.MonkeyPatch) -> None:
    predictor_module = pytest.importorskip("vidur.execution_time_predictor.sklearn_execution_time_predictor")
    SklearnExecutionTimePredictor = predictor_module.SklearnExecutionTimePredictor

    df = pd.DataFrame(
        {
            "num_tokens": [128, 256, 512],
            "time_stats.op.median": [1.0, None, 3.0],
        }
    )

    def _fake_train_model(self, model_name, df, feature_cols, target_col):  # type: ignore[no-untyped-def]
        assert int(len(df)) == 2
        assert not df[str(target_col)].isna().any()
        return {"trained": True}

    monkeypatch.setattr(SklearnExecutionTimePredictor, "_train_model", _fake_train_model, raising=True)

    summary: dict[str, object] = {}
    with patch_vidur_sklearn_train_model_dropna(summary=summary):
        result = SklearnExecutionTimePredictor._train_model(  # type: ignore[misc]
            object(),
            model_name="op.median",
            df=df,
            feature_cols=["num_tokens"],
            target_col="time_stats.op.median",
        )

    assert result == {"trained": True}
    assert isinstance(summary.get("per_model"), dict)
    per_model = summary["per_model"]
    assert isinstance(per_model, dict)
    stats = per_model["op.median"]
    assert stats["rows_total"] == 3
    assert stats["rows_used"] == 2
    assert stats["rows_dropped"] == 1


def test_patch_vidur_sklearn_train_model_dropna_raises_on_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    predictor_module = pytest.importorskip("vidur.execution_time_predictor.sklearn_execution_time_predictor")
    SklearnExecutionTimePredictor = predictor_module.SklearnExecutionTimePredictor

    df = pd.DataFrame({"num_tokens": [128, 256], "time_stats.op.median": [None, None]})

    def _fake_train_model(self, model_name, df, feature_cols, target_col):  # type: ignore[no-untyped-def]
        raise AssertionError("Original train_model should not be called when filtered df is empty.")

    monkeypatch.setattr(SklearnExecutionTimePredictor, "_train_model", _fake_train_model, raising=True)

    summary: dict[str, object] = {}
    with patch_vidur_sklearn_train_model_dropna(summary=summary):
        with pytest.raises(RuntimeError, match="no usable rows"):
            SklearnExecutionTimePredictor._train_model(  # type: ignore[misc]
                object(),
                model_name="op.median",
                df=df,
                feature_cols=["num_tokens"],
                target_col="time_stats.op.median",
            )
