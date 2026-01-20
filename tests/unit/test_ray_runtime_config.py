"""
Unit tests for Ray runtime env/config precedence and reporting.

These tests validate the behavior of `gpu_simulate_test.ray_runtime.apply_ray_env_defaults`.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from gpu_simulate_test.ray_runtime import SUPPORTED_RAY_ENV_KEYS, apply_ray_env_defaults
from gpu_simulate_test.vidur_cli.errors import UserFacingError


def test_config_injects_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config values are injected into env vars when env is unset."""

    for key in SUPPORTED_RAY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    cfg: dict[str, Any] = {
        "RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES": 4000000000,
        "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION": 0.10,
        "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE": True,
    }
    settings = apply_ray_env_defaults(cfg)

    assert os.environ["RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES"] == "4000000000"
    assert os.environ["RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION"] == "0.1"
    assert os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] == "1"
    assert {setting.source for setting in settings} == {"configuration"}


def test_none_values_are_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Null config values are treated as "leave to Ray defaults" (no env injection)."""

    for key in SUPPORTED_RAY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    cfg = {key: None for key in SUPPORTED_RAY_ENV_KEYS}
    settings = apply_ray_env_defaults(cfg)

    for key in SUPPORTED_RAY_ENV_KEYS:
        assert key not in os.environ

    assert all(setting.source == "default" for setting in settings)
    assert all(setting.effective_value is None for setting in settings)


def test_env_wins_over_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a supported env var is set, config must not override it."""

    for key in SUPPORTED_RAY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES", "123")
    cfg: dict[str, Any] = {"RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES": 999}
    settings = apply_ray_env_defaults(cfg)

    assert os.environ["RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES"] == "123"
    sources = {setting.key: setting.source for setting in settings}
    assert sources["RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES"] == "environment"


def test_mixed_sourcing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some settings may come from env and others from config (per-key precedence)."""

    for key in SUPPORTED_RAY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES", "123")
    cfg: dict[str, Any] = {
        "RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES": 999,
        "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION": 0.1,
    }
    settings = apply_ray_env_defaults(cfg)

    sources = {setting.key: setting.source for setting in settings}
    assert sources["RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES"] == "environment"
    assert sources["RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION"] == "configuration"


def test_invalid_config_values_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid config values fail fast with actionable errors."""

    for key in SUPPORTED_RAY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(UserFacingError, match="RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES"):
        apply_ray_env_defaults({"RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES": -1})

    with pytest.raises(UserFacingError, match="RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION"):
        apply_ray_env_defaults({"RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION": 0})

    with pytest.raises(UserFacingError, match="RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"):
        apply_ray_env_defaults({"RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE": "yes"})


def test_invalid_env_values_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid env values fail fast before starting Ray."""

    for key in SUPPORTED_RAY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES", "")
    with pytest.raises(UserFacingError, match="RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES"):
        apply_ray_env_defaults(None)

    monkeypatch.delenv("RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES", raising=False)
    monkeypatch.setenv("RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION", "2.0")
    with pytest.raises(UserFacingError, match="RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION"):
        apply_ray_env_defaults(None)

    monkeypatch.delenv("RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION", raising=False)
    monkeypatch.setenv("RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE", "maybe")
    with pytest.raises(UserFacingError, match="RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"):
        apply_ray_env_defaults(None)


def test_unsupported_config_keys_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown keys under ray.env are rejected."""

    for key in SUPPORTED_RAY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(UserFacingError, match="Unsupported Ray setting"):
        apply_ray_env_defaults({"RAY_SOMETHING_UNSUPPORTED": 123})
