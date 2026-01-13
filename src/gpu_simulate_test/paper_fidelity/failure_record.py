from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, cast

from gpu_simulate_test.io import utcnow_iso, write_json

BlockerCategory = Literal[
    "insufficient GPUs",
    "OOM",
    "missing model files",
    "unsupported model",
    "unknown",
]

FailureAction = Literal["trace", "profile", "repro", "matrix"]

Workload = Literal["static", "dynamic"]
Scale = Literal["small", "medium", "full"]


@dataclass(frozen=True)
class FailureRecord:
    schema_version: str
    generated_at: str
    run_id: str
    action: FailureAction
    scenario_key: str
    scenario_name: str | None
    workload: Workload | None
    scale: Scale | None
    attempted_command: list[str] | str | None
    hydra_overrides: list[str]
    error_message: str
    traceback: str | None
    blocker_category: BlockerCategory

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "run_id": self.run_id,
            "action": self.action,
            "scenario_key": self.scenario_key,
            "scenario_name": self.scenario_name,
            "workload": self.workload,
            "scale": self.scale,
            "attempted_command": self.attempted_command,
            "hydra_overrides": self.hydra_overrides,
            "error_message": self.error_message,
            "traceback": self.traceback,
            "blocker_category": self.blocker_category,
        }


def write_failure_record(out_path: Path, record: FailureRecord) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, record.to_dict())
    return out_path.resolve()


def build_failure_record(
    *,
    run_id: str,
    action: FailureAction,
    scenario_key: str,
    scenario_name: str | None,
    workload: str | None,
    scale: str | None,
    attempted_command: Sequence[str] | str | None,
    hydra_overrides: Sequence[str] | None,
    error_message: str,
    traceback: str | None,
    blocker_category: BlockerCategory,
) -> FailureRecord:
    if attempted_command is None or isinstance(attempted_command, str):
        attempted_command_value = attempted_command
    else:
        attempted_command_value = [str(x) for x in attempted_command]

    return FailureRecord(
        schema_version="v1",
        generated_at=utcnow_iso(),
        run_id=str(run_id),
        action=action,
        scenario_key=str(scenario_key),
        scenario_name=None if scenario_name is None else str(scenario_name),
        workload=None if workload is None else cast(Workload, str(workload)),
        scale=None if scale is None else cast(Scale, str(scale)),
        attempted_command=attempted_command_value,
        hydra_overrides=[str(x) for x in (hydra_overrides or [])],
        error_message=str(error_message),
        traceback=None if traceback is None else str(traceback),
        blocker_category=blocker_category,
    )


def categorize_blocker(*, error_message: str, traceback: str | None = None) -> BlockerCategory:
    text = "\n".join([error_message, traceback or ""]).lower()

    if "insufficient gpus" in text or ("required=" in text and "available=" in text and "gpu" in text):
        return "insufficient GPUs"

    if "cuda out of memory" in text or "out of memory" in text or "oom" in text:
        return "OOM"

    if "missing model assets" in text or "no such file or directory" in text:
        return "missing model files"

    if "unsupported" in text and "model" in text:
        return "unsupported model"

    return "unknown"
