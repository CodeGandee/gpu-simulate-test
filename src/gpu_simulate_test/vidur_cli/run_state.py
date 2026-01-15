"""
Run state persistence for `vidur-cli`.

This module defines helpers for the v1 run directory artifacts:
- `run_state.json` (stage state machine)
- `failure.json` (most recent failure metadata)

The JSON schemas for these artifacts live under `specs/004-vidur-cli/contracts/`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Literal, TypeVar

from gpu_simulate_test.io import read_json, utcnow_iso, write_json
from gpu_simulate_test.vidur_cli.errors import UserFacingError


StageName = Literal["init-run", "trace", "profile", "sim", "real", "report", "resources", "configs"]


@dataclass(frozen=True)
class Presets:
    model: str
    hardware: str
    backend: str
    workload: str
    vidur: str


def utc_timestamp_tag() -> str:
    """Return a filesystem-friendly UTC timestamp tag."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sanitize_tag(value: str) -> str:
    """Make a filesystem-safe tag (keep letters/digits/._+-)."""
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9._+-]+", "_", value)
    return value.strip("_") or "run"


def default_run_tag(presets: Presets) -> str:
    """Default run tag format: `preset+timestamp` (UTC)."""
    parts = [
        f"m={presets.model}",
        f"h={presets.hardware}",
        f"b={presets.backend}",
        f"w={presets.workload}",
        f"v={presets.vidur}",
        utc_timestamp_tag(),
    ]
    return sanitize_tag("+".join(parts))


def normalize_run_dir(*, run_dir: str, workspace_root: Path) -> Path:
    """Normalize a `--run-dir` argument.

    Relative paths are interpreted relative to `workspace_root` (per spec FR-015).
    """
    p = Path(run_dir).expanduser()
    if not p.is_absolute():
        p = workspace_root / p
    return p.resolve()


def load_run_state(*, run_dir: Path) -> dict[str, Any]:
    """Load `<run_dir>/run_state.json`."""
    path = run_dir / "run_state.json"
    if not path.exists():
        raise UserFacingError(
            f"Missing prerequisite: run_state.json ({path})",
            hint="Run `vidur-cli svr init-run ...` first to create a run directory.",
        )
    data = read_json(path)
    if data.get("schema_version") != "v1":
        raise UserFacingError(
            f"{path}: unsupported schema_version (expected 'v1')",
            hint="Recreate the run directory with the current version of vidur-cli.",
            context={"schema_version": data.get("schema_version")},
        )
    return data


def write_run_state(*, run_dir: Path, presets: Presets, overrides: list[str], run_tag: str | None = None) -> Path:
    """Create and write `<run_dir>/run_state.json` (schema v1)."""
    run_dir = run_dir.expanduser().resolve()
    tag = run_tag or run_dir.name
    created_at = utcnow_iso()
    payload: dict[str, Any] = {
        "schema_version": "v1",
        "created_at": created_at,
        "updated_at": created_at,
        "run_tag": str(tag),
        "run_dir": str(run_dir),
        "presets": {
            "model": presets.model,
            "hardware": presets.hardware,
            "backend": presets.backend,
            "workload": presets.workload,
            "vidur": presets.vidur,
        },
        "overrides": list(overrides),
        "artifacts": {},
    }
    out = run_dir / "run_state.json"
    write_json(out, payload)
    return out.resolve()


def save_run_state(*, run_dir: Path, run_state: Mapping[str, Any]) -> Path:
    """Persist an updated run state, refreshing `updated_at`."""
    payload = dict(run_state)
    payload["updated_at"] = utcnow_iso()
    out = run_dir / "run_state.json"
    write_json(out, payload)
    return out.resolve()


def write_failure_json(
    *,
    run_dir: Path,
    stage: StageName,
    error_type: str,
    message: str,
    context: Mapping[str, Any] | None,
) -> Path:
    """Write `<run_dir>/failure.json` (schema v1) and return the absolute path."""
    payload: dict[str, Any] = {
        "schema_version": "v1",
        "failed_at": utcnow_iso(),
        "stage": stage,
        "error_type": str(error_type),
        "message": str(message),
    }
    if context:
        payload["context"] = dict(context)
    out = run_dir / "failure.json"
    write_json(out, payload)
    return out.resolve()


T = TypeVar("T")


def run_with_failure_json(
    *,
    run_dir: Path,
    stage: StageName,
    fn: Callable[[], T],
    failure_context: Mapping[str, Any] | None = None,
    on_error: Callable[[dict[str, Any], str], dict[str, Any]] | None = None,
) -> T:
    """Run a stage function, writing `failure.json` on exceptions.

    Parameters
    ----------
    run_dir
        Run directory to write failure metadata into.
    stage
        Stage name (recorded in `failure.json`).
    fn
        Stage function to execute.
    failure_context
        Optional extra failure context merged into the failure payload.
    on_error
        Optional callback to update `run_state.json` after a failure. Receives
        `(run_state, ended_at)` and must return the updated run_state.
    """
    try:
        return fn()
    except Exception as e:
        ctx: dict[str, Any] = dict(failure_context) if failure_context else {}
        if isinstance(e, UserFacingError):
            message = e.message
            if e.hint:
                ctx.setdefault("hint", e.hint)
            if e.context:
                ctx.setdefault("context", e.context)
        else:
            message = str(e)

        ended_at = utcnow_iso()
        write_failure_json(
            run_dir=run_dir,
            stage=stage,
            error_type=type(e).__name__,
            message=message,
            context=ctx or None,
        )

        if on_error is not None:
            try:
                state = load_run_state(run_dir=run_dir)
                updated = on_error(state, ended_at)
                save_run_state(run_dir=run_dir, run_state=updated)
            except Exception:
                pass

        raise


def require_file(path: Path, *, what: str) -> Path:
    """Require a file to exist, raising a user-facing error if missing."""
    if not path.exists():
        raise UserFacingError(
            f"Missing prerequisite: {what} ({path})",
            hint="Run the prerequisite stage first (see `vidur-cli --help`).",
        )
    if not path.is_file():
        raise UserFacingError(f"Expected a file for {what}, got: {path}")
    return path


def require_dir(path: Path, *, what: str) -> Path:
    """Require a directory to exist, raising a user-facing error if missing."""
    if not path.exists():
        raise UserFacingError(
            f"Missing prerequisite: {what} ({path})",
            hint="Run the prerequisite stage first (see `vidur-cli --help`).",
        )
    if not path.is_dir():
        raise UserFacingError(f"Expected a directory for {what}, got: {path}")
    return path


@dataclass(frozen=True)
class TraceArtifacts:
    trace_csv: Path
    trace_meta_json: Path
    trace_lengths_csv: Path
    trace_intervals_csv: Path


def require_trace_artifacts(*, run_dir: Path) -> TraceArtifacts:
    """Require `svr trace` outputs to exist."""
    state = load_run_state(run_dir=run_dir)
    artifacts = state.get("artifacts") or {}
    trace = artifacts.get("trace")
    if not isinstance(trace, dict):
        raise UserFacingError(
            "Missing prerequisite: trace artifacts are not recorded in run_state.json",
            hint="Run `vidur-cli svr trace --run-dir <run_dir>` first.",
        )

    trace_csv = Path(str(trace.get("trace_csv", run_dir / "trace" / "trace.csv"))).expanduser()
    trace_meta = Path(str(trace.get("trace_meta_json", run_dir / "trace" / "trace_meta.json"))).expanduser()
    trace_lengths = Path(str(trace.get("trace_lengths_csv", run_dir / "trace" / "trace_lengths.csv"))).expanduser()
    trace_intervals = Path(str(trace.get("trace_intervals_csv", run_dir / "trace" / "trace_intervals.csv"))).expanduser()

    require_file(trace_csv, what="trace/trace.csv")
    require_file(trace_meta, what="trace/trace_meta.json")
    require_file(trace_lengths, what="trace/trace_lengths.csv")
    require_file(trace_intervals, what="trace/trace_intervals.csv")
    return TraceArtifacts(
        trace_csv=trace_csv.resolve(),
        trace_meta_json=trace_meta.resolve(),
        trace_lengths_csv=trace_lengths.resolve(),
        trace_intervals_csv=trace_intervals.resolve(),
    )


@dataclass(frozen=True)
class ProfileArtifacts:
    profiling_root: Path
    include_cpu_overhead: bool


def require_profile_artifacts(*, run_dir: Path) -> ProfileArtifacts:
    """Require `svr profile` outputs to exist."""
    state = load_run_state(run_dir=run_dir)
    artifacts = state.get("artifacts") or {}
    profile = artifacts.get("profile")
    if not isinstance(profile, dict):
        raise UserFacingError(
            "Missing prerequisite: profile artifacts are not recorded in run_state.json",
            hint="Run `vidur-cli svr profile --run-dir <run_dir>` first.",
        )
    profiling_root = Path(str(profile.get("profiling_root", run_dir / "profile"))).expanduser()
    require_dir(profiling_root, what="profiling root")
    include_cpu_overhead = bool(profile.get("include_cpu_overhead", True))
    return ProfileArtifacts(profiling_root=profiling_root.resolve(), include_cpu_overhead=include_cpu_overhead)


@dataclass(frozen=True)
class SimArtifacts:
    sim_run_dir: Path


def require_sim_artifacts(*, run_dir: Path) -> SimArtifacts:
    """Require `svr sim` outputs to exist."""
    state = load_run_state(run_dir=run_dir)
    artifacts = state.get("artifacts") or {}
    sim = artifacts.get("sim")
    if not isinstance(sim, dict):
        raise UserFacingError(
            "Missing prerequisite: sim artifacts are not recorded in run_state.json",
            hint="Run `vidur-cli svr sim --run-dir <run_dir>` first.",
        )
    sim_run_dir = Path(str(sim.get("sim_run_dir", run_dir / "sim"))).expanduser()
    require_dir(sim_run_dir, what="sim_run_dir")
    return SimArtifacts(sim_run_dir=sim_run_dir.resolve())


@dataclass(frozen=True)
class RealArtifacts:
    real_run_dir: Path
    backend: str


def require_real_artifacts(*, run_dir: Path) -> RealArtifacts:
    """Require `svr real` outputs to exist."""
    state = load_run_state(run_dir=run_dir)
    artifacts = state.get("artifacts") or {}
    real = artifacts.get("real")
    if not isinstance(real, dict):
        raise UserFacingError(
            "Missing prerequisite: real artifacts are not recorded in run_state.json",
            hint="Run `vidur-cli svr real --run-dir <run_dir>` first.",
        )
    real_run_dir = Path(str(real.get("real_run_dir", run_dir / "real"))).expanduser()
    require_dir(real_run_dir, what="real_run_dir")
    backend = str(real.get("backend", "unknown"))
    return RealArtifacts(real_run_dir=real_run_dir.resolve(), backend=backend)
