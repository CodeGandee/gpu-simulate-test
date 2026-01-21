from __future__ import annotations

import os

from gpu_simulate_test.env_guard import (
    DEFAULT_RAY_OBJECT_STORE_MAX_MEMORY_BYTES_CAP,
    RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES_ENV,
    apply_ray_object_store_defaults,
)


def test_apply_ray_object_store_defaults_sets_cap_when_unset(monkeypatch) -> None:
    monkeypatch.delenv(RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES_ENV, raising=False)
    apply_ray_object_store_defaults()
    value = os.environ.get(RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES_ENV)
    assert value is not None
    assert int(value) > 0
    assert int(value) <= DEFAULT_RAY_OBJECT_STORE_MAX_MEMORY_BYTES_CAP


def test_apply_ray_object_store_defaults_does_not_override_user_value(monkeypatch) -> None:
    monkeypatch.setenv(RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES_ENV, "123")
    apply_ray_object_store_defaults()
    assert os.environ[RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES_ENV] == "123"

