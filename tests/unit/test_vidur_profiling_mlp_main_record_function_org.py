from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from gpu_simulate_test.vidur_ext.record_function_tracer_v2 import RecordFunctionTracerV2
from gpu_simulate_test.vidur_ext.vidur_profiling_mlp_main import main as wrapper_main


def _install_fake_vidur_modules(*, monkeypatch: pytest.MonkeyPatch, captured_argv: list[list[str]]) -> types.ModuleType:
    """Install a minimal fake `vidur.profiling.mlp` module tree into sys.modules.

    This avoids importing the real Vidur stack in unit tests while still exercising:
    - argv rewriting
    - tracer patch gating
    """
    vidur_mod = types.ModuleType("vidur")
    vidur_mod.__path__ = []  # type: ignore[attr-defined]

    profiling_mod = types.ModuleType("vidur.profiling")
    profiling_mod.__path__ = []  # type: ignore[attr-defined]

    mlp_pkg = types.ModuleType("vidur.profiling.mlp")
    mlp_pkg.__path__ = []  # type: ignore[attr-defined]

    mlp_wrapper_mod = types.ModuleType("vidur.profiling.mlp.mlp_wrapper")
    sentinel_tracer = object()
    mlp_wrapper_mod.RecordFunctionTracer = sentinel_tracer  # type: ignore[attr-defined]

    main_mod = types.ModuleType("vidur.profiling.mlp.main")

    def _fake_vidur_mlp_main() -> None:
        captured_argv.append(list(sys.argv))

    main_mod.main = _fake_vidur_mlp_main  # type: ignore[attr-defined]

    vidur_mod.profiling = profiling_mod  # type: ignore[attr-defined]
    profiling_mod.mlp = mlp_pkg  # type: ignore[attr-defined]
    mlp_pkg.mlp_wrapper = mlp_wrapper_mod  # type: ignore[attr-defined]
    mlp_pkg.main = main_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "vidur", vidur_mod)
    monkeypatch.setitem(sys.modules, "vidur.profiling", profiling_mod)
    monkeypatch.setitem(sys.modules, "vidur.profiling.mlp", mlp_pkg)
    monkeypatch.setitem(sys.modules, "vidur.profiling.mlp.mlp_wrapper", mlp_wrapper_mod)
    monkeypatch.setitem(sys.modules, "vidur.profiling.mlp.main", main_mod)
    return mlp_wrapper_mod


def test_record_function_patches_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GSIM_CUDA_VISIBLE_DEVICES", "0")

    captured: list[list[str]] = []
    mlp_wrapper_mod = _install_fake_vidur_modules(monkeypatch=monkeypatch, captured_argv=captured)

    monkeypatch.setattr(
        sys,
        "argv",
        ["vidur_profiling_mlp_main", "--profile_method", "record_function"],
        raising=False,
    )
    wrapper_main()

    assert captured, "expected fake Vidur main to be invoked"
    assert captured[0][1:] == ["--profile_method", "record_function"]
    assert getattr(mlp_wrapper_mod, "RecordFunctionTracer") is RecordFunctionTracerV2


def test_record_function_org_does_not_patch_and_rewrites(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GSIM_CUDA_VISIBLE_DEVICES", "0")

    captured: list[list[str]] = []
    mlp_wrapper_mod = _install_fake_vidur_modules(monkeypatch=monkeypatch, captured_argv=captured)
    original_tracer: Any = getattr(mlp_wrapper_mod, "RecordFunctionTracer")

    monkeypatch.setattr(
        sys,
        "argv",
        ["vidur_profiling_mlp_main", "--profile_method", "record_function_org"],
        raising=False,
    )
    wrapper_main()

    assert captured, "expected fake Vidur main to be invoked"
    assert captured[0][1:] == ["--profile_method", "record_function"]
    assert getattr(mlp_wrapper_mod, "RecordFunctionTracer") is original_tracer

