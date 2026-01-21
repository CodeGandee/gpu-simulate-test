from __future__ import annotations

import os
import warnings

import pytest

from gpu_simulate_test.env_guard import (
    RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES_ENV,
    warn_if_ray_object_store_unconfigured,
)


def test_warn_if_ray_object_store_unconfigured_emits_warning_when_unset(monkeypatch) -> None:
    monkeypatch.delenv(RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES_ENV, raising=False)
    with pytest.warns(RuntimeWarning):
        warn_if_ray_object_store_unconfigured()
    assert os.environ.get(RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES_ENV) is None


def test_warn_if_ray_object_store_unconfigured_no_warning_when_set(monkeypatch) -> None:
    monkeypatch.setenv(RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES_ENV, "123")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warn_if_ray_object_store_unconfigured()
    assert len(record) == 0
