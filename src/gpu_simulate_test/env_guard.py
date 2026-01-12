from __future__ import annotations

import os
from pathlib import Path


GSIM_CUDA_VISIBLE_DEVICES_ENV = "GSIM_CUDA_VISIBLE_DEVICES"
RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO_ENV = "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"


def find_repo_root(start: Path | None = None) -> Path | None:
    """Best-effort repo root discovery (looks for `pyproject.toml`).

    Hydra often changes the working directory (e.g. into `tmp/...`). We prefer a marker-based
    search instead of relying on `cwd`.
    """
    starts: list[Path] = []
    if start is not None:
        starts.append(start)
    starts.append(Path.cwd())
    starts.append(Path(__file__).resolve())

    seen: set[Path] = set()
    for candidate in starts:
        current = candidate if candidate.is_dir() else candidate.parent
        for parent in [current, *current.parents]:
            if parent in seen:
                continue
            seen.add(parent)
            if (parent / "pyproject.toml").exists():
                return parent
    return None


def load_dotenv_if_present(*, repo_root: Path | None = None) -> Path | None:
    """Load `.env` (if present) into `os.environ` without overriding existing keys.

    Returns the resolved `.env` path when loaded, otherwise `None`.
    """
    if repo_root is None:
        repo_root = find_repo_root()
    if repo_root is None:
        return None

    dotenv = (repo_root / ".env").resolve()
    if not dotenv.exists():
        return None

    for line in dotenv.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.startswith("export "):
            raw = raw[len("export ") :].lstrip()
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if (len(value) >= 2) and (
            (value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")
        ):
            value = value[1:-1]
        os.environ[key] = value
    return dotenv


def require_gsim_cuda_visible_devices() -> str:
    """Return `GSIM_CUDA_VISIBLE_DEVICES`, raising if missing/empty."""
    value = os.environ.get(GSIM_CUDA_VISIBLE_DEVICES_ENV)
    if value is None or not value.strip():
        raise RuntimeError(
            "Refusing to run GPU work without an explicit GPU pin. "
            f"Set `{GSIM_CUDA_VISIBLE_DEVICES_ENV}` (e.g. `0` or `0,1`). "
            "Tip: you can define it in a repo-local `.env` file."
        )
    return value.strip()


def apply_cuda_visible_devices_from_gsim(*, repo_root: Path | None = None) -> str:
    """Load `.env` (if present), require GSIM pinning, and set `CUDA_VISIBLE_DEVICES`.

    Returns the CUDA-visible devices string that was applied.
    """
    load_dotenv_if_present(repo_root=repo_root)
    desired = require_gsim_cuda_visible_devices()
    # Normalize whitespace but otherwise preserve user intent (CUDA accepts comma-separated lists).
    desired = ",".join(part.strip() for part in desired.split(",") if part.strip())
    if not desired:
        raise RuntimeError(
            f"`{GSIM_CUDA_VISIBLE_DEVICES_ENV}` is set but empty after normalization; "
            "expected a non-empty CUDA_VISIBLE_DEVICES-style value (e.g. `0` or `0,1`)."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = desired
    # Ray currently overrides `CUDA_VISIBLE_DEVICES` to an empty string for actors with `num_gpus=0`
    # (which Sarathi uses), which can make GPU work fail inside Ray workers. This knob tells Ray to
    # preserve the existing env var instead. The warning emitted by Ray suggests this exact setting.
    os.environ.setdefault(RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO_ENV, "0")
    return desired


def patch_sarathi_preserve_cuda_visible_devices() -> None:
    """Prevent Sarathi from unsetting CUDA_VISIBLE_DEVICES in Ray workers.

    Sarathi's `sarathi.engine.base_llm_engine.RayWorker` clears CUDA_VISIBLE_DEVICES after Ray sets it,
    which can expose unusable GPUs on some hosts and defeat explicit pinning. This patch keeps
    CUDA_VISIBLE_DEVICES intact so that Ray's per-actor assignment (or our global pin) remains active.
    """
    def _noop() -> None:
        return None

    # Patch all call-sites we know about:
    # - sarathi.utils.unset_cuda_visible_devices (source of truth)
    # - copies imported into sarathi.engine.base_llm_engine and sarathi.engine.ray_utils
    #
    # Each import can fail if sarathi (or torch) is unavailable; treat as best-effort.
    try:
        from sarathi import utils as _sarathi_utils  # type: ignore

        _sarathi_utils.unset_cuda_visible_devices = _noop  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        from sarathi.engine import base_llm_engine as _base_llm_engine  # type: ignore

        _base_llm_engine.unset_cuda_visible_devices = _noop  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        from sarathi.engine import ray_utils as _ray_utils  # type: ignore

        _ray_utils.unset_cuda_visible_devices = _noop  # type: ignore[attr-defined]
    except Exception:
        pass
