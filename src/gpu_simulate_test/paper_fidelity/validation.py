from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


class ScenarioPreflightError(RuntimeError):
    """Base class for all scenario preflight failures."""


class MissingModelFilesError(ScenarioPreflightError):
    pass


class MissingTraceSourceError(ScenarioPreflightError):
    pass


class InsufficientGpusError(ScenarioPreflightError):
    def __init__(self, *, required: int, available: int) -> None:
        super().__init__(f"Insufficient GPUs: required={required} available={available}")
        self.required = int(required)
        self.available = int(available)


@dataclass(frozen=True)
class ScenarioRequirements:
    scenario_key: str
    scenario_name: str
    model_ref: Path
    trace_source: Path
    required_gpus: int


def required_gpus_from_cfg(cfg: DictConfig) -> int:
    tp = int(OmegaConf.select(cfg, "scenario.real.parallel.tensor_parallel_size") or 1)
    pp = int(OmegaConf.select(cfg, "scenario.real.parallel.pipeline_parallel_size") or 1)
    if tp < 1 or pp < 1:
        raise ScenarioPreflightError(f"Invalid parallelism: tp={tp} pp={pp} (both must be >= 1)")
    return tp * pp


def _resolve_repo_path(value: str | None, *, repo_root: Path) -> Path:
    if not value:
        raise ScenarioPreflightError("Missing required path in scenario config")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()
    return path


def preflight_common(cfg: DictConfig, *, repo_root: Path) -> ScenarioRequirements:
    scenario_name = str(OmegaConf.select(cfg, "scenario.name") or "unknown")

    model_ref = _resolve_repo_path(
        OmegaConf.select(cfg, "scenario.model.model_ref"),
        repo_root=repo_root,
    )
    trace_source = _resolve_repo_path(
        OmegaConf.select(cfg, "scenario.trace_source.path"),
        repo_root=repo_root,
    )

    return ScenarioRequirements(
        scenario_key=scenario_name,
        scenario_name=scenario_name,
        model_ref=model_ref,
        trace_source=trace_source,
        required_gpus=required_gpus_from_cfg(cfg),
    )


def preflight_trace(cfg: DictConfig, *, repo_root: Path) -> None:
    req = preflight_common(cfg, repo_root=repo_root)
    if not req.trace_source.exists():
        raise MissingTraceSourceError(f"Missing trace source: {req.trace_source}")


def preflight_profile(cfg: DictConfig, *, repo_root: Path, available_gpus: int) -> None:
    req = preflight_common(cfg, repo_root=repo_root)
    if not req.model_ref.exists():
        raise MissingModelFilesError(f"Missing model assets: {req.model_ref}")
    if available_gpus and available_gpus < req.required_gpus:
        raise InsufficientGpusError(required=req.required_gpus, available=available_gpus)


def preflight_repro(
    cfg: DictConfig,
    *,
    repo_root: Path,
    available_gpus: int,
) -> None:
    req = preflight_common(cfg, repo_root=repo_root)
    if not req.model_ref.exists():
        raise MissingModelFilesError(f"Missing model assets: {req.model_ref}")
    if not req.trace_source.exists():
        raise MissingTraceSourceError(f"Missing trace source: {req.trace_source}")
    if available_gpus and available_gpus < req.required_gpus:
        raise InsufficientGpusError(required=req.required_gpus, available=available_gpus)
