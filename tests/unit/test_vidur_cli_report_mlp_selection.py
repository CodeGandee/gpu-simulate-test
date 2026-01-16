from __future__ import annotations

import json
from pathlib import Path

from gpu_simulate_test.vidur_cli.reporting import _load_profile_mlp_selection


def _write_run_state(run_dir: Path, *, artifacts: dict) -> None:
    run_state = {"schema_version": "v1", "run_dir": str(run_dir), "artifacts": artifacts}
    (run_dir / "run_state.json").write_text(json.dumps(run_state), encoding="utf-8")


def test_load_profile_mlp_selection_prefers_structured_payload(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _write_run_state(
        run_dir,
        artifacts={
            "profile": {
                "status": "ok",
                "profiling_root": str(run_dir / "profile"),
                "include_cpu_overhead": True,
                "overrides": [],
                "mlp": {"profile_method": "cuda_event", "requested_profile_method": "cuda_event"},
            }
        },
    )
    assert _load_profile_mlp_selection(run_dir=run_dir) == {
        "profile_method": "cuda_event",
        "requested_profile_method": "cuda_event",
    }


def test_load_profile_mlp_selection_falls_back_to_overrides(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _write_run_state(
        run_dir,
        artifacts={
            "profile": {
                "status": "ok",
                "profiling_root": str(run_dir / "profile"),
                "include_cpu_overhead": True,
                "overrides": [
                    "profiling.mlp.profile_method=record_function",
                    "profiling.mlp.validation.mode=non_strict",
                    "profiling.mlp.fallback.enabled=true",
                    "profiling.mlp.fallback.method=cuda_event",
                ],
            }
        },
    )
    selection = _load_profile_mlp_selection(run_dir=run_dir)
    assert selection is not None
    assert selection["requested_profile_method"] == "record_function"
    assert selection["profile_method"] == "record_function"
    assert selection["validation_mode"] == "non_strict"
    assert selection["fallback_enabled"] is True
    assert selection["fallback_method"] == "cuda_event"


def test_load_profile_mlp_selection_falls_back_to_staging_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _write_run_state(
        run_dir,
        artifacts={
            "profile": {
                "status": "ok",
                "profiling_root": str(run_dir / "profile"),
                "include_cpu_overhead": True,
                "overrides": [],
            }
        },
    )

    config_dir = run_dir / "profile" / "_staging" / "mlp" / "2026-01-16_00-00-00"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yaml").write_text("profile_method: cuda_event\n", encoding="utf-8")

    assert _load_profile_mlp_selection(run_dir=run_dir) == {
        "requested_profile_method": "cuda_event",
        "profile_method": "cuda_event",
    }

