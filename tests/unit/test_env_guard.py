from __future__ import annotations

import os
from pathlib import Path

import pytest

from gpu_simulate_test.env_guard import apply_cuda_visible_devices_from_gsim, load_dotenv_if_present


def test_apply_cuda_visible_devices_from_gsim_uses_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GSIM_CUDA_VISIBLE_DEVICES", "0, 1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    applied = apply_cuda_visible_devices_from_gsim(repo_root=tmp_path)

    assert applied == "0,1"
    assert applied == (os.environ.get("CUDA_VISIBLE_DEVICES") or "")


def test_apply_cuda_visible_devices_from_gsim_loads_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("GSIM_CUDA_VISIBLE_DEVICES=2\n", encoding="utf-8")
    monkeypatch.delenv("GSIM_CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    applied = apply_cuda_visible_devices_from_gsim(repo_root=tmp_path)

    assert applied == "2"
    assert applied == (os.environ.get("CUDA_VISIBLE_DEVICES") or "")


def test_apply_cuda_visible_devices_from_gsim_requires_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GSIM_CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(RuntimeError, match="GSIM_CUDA_VISIBLE_DEVICES"):
        apply_cuda_visible_devices_from_gsim(repo_root=tmp_path)


def test_load_dotenv_if_present_does_not_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("FOO=from_dotenv\n", encoding="utf-8")
    monkeypatch.setenv("FOO", "from_env")

    load_dotenv_if_present(repo_root=tmp_path)

    assert (os.environ.get("FOO") or "") == "from_env"
