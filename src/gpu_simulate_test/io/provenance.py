from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class GitInfo:
    commit: str | None
    dirty: bool | None


def get_git_info(*, repo_root: Path) -> GitInfo:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit = None

    try:
        status = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root)
            .decode("utf-8")
            .strip()
        )
        dirty = bool(status)
    except Exception:
        dirty = None

    return GitInfo(commit=commit, dirty=dirty)


def build_env_snapshot() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }
    try:
        import torch  # type: ignore

        env["torch"] = getattr(torch, "__version__", None)
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return env

