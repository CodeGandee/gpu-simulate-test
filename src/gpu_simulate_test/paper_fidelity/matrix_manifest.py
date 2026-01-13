from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from gpu_simulate_test.io import build_env_snapshot, get_git_info, utcnow_iso, write_json

RunStatus = Literal["success", "failure"]


@dataclass(frozen=True)
class MatrixRunEntry:
    scenario_key: str
    workload: Literal["static", "dynamic"]
    scale: Literal["small", "medium", "full"]
    status: RunStatus
    report_dir: Path | None = None
    failure_record_json: Path | None = None
    blocker_category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_key": self.scenario_key,
            "workload": self.workload,
            "scale": self.scale,
            "status": self.status,
            "report_dir": None if self.report_dir is None else str(self.report_dir.resolve()),
            "failure_record_json": None
            if self.failure_record_json is None
            else str(self.failure_record_json.resolve()),
            "blocker_category": self.blocker_category,
        }


def write_matrix_manifest(
    *,
    out_path: Path,
    repo_root: Path,
    run_id: str,
    scenarios: list[str],
    workloads: list[str],
    scale: str,
    runs: list[MatrixRunEntry],
) -> Path:
    git = get_git_info(repo_root=repo_root)
    payload: dict[str, Any] = {
        "schema_version": "v1",
        "generated_at": utcnow_iso(),
        "run_id": str(run_id),
        "scenarios": list(scenarios),
        "workloads": list(workloads),
        "scale": str(scale),
        "git_commit": git.commit or "unknown",
        "git_dirty": git.dirty,
        "env": build_env_snapshot(),
        "runs": [r.to_dict() for r in runs],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, payload)
    return out_path.resolve()

