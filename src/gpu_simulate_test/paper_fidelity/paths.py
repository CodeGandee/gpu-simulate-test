"""
Path conventions and run metadata helpers for paper-fidelity workflows.

All heavy artifacts are written under `tmp/paper_fidelity/`; human-readable reports are written
under `results/reports/<date>/paper_fidelity/<scenario>/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from gpu_simulate_test.io import build_env_snapshot, get_git_info, utcnow_iso


@dataclass(frozen=True)
class PaperFidelityPaths:
    """Stable locations for all paper-fidelity artifacts (no DB)."""

    repo_root: Path

    @property
    def tmp_root(self) -> Path:
        return self.repo_root / "tmp" / "paper_fidelity"

    @property
    def results_root(self) -> Path:
        return self.repo_root / "results"

    def trace_dir(self, scenario_name: str) -> Path:
        return self.tmp_root / "traces" / scenario_name

    def sim_dir(self, scenario_name: str) -> Path:
        return self.tmp_root / "runs" / scenario_name / "sim"

    def real_dir(self, scenario_name: str) -> Path:
        return self.tmp_root / "runs" / scenario_name / "real"

    def capacity_dir(self, scenario_name: str) -> Path:
        return self.tmp_root / "runs" / scenario_name / "capacity"

    def reports_dir(self, *, date: str, scenario_name: str) -> Path:
        return self.results_root / "reports" / date / "paper_fidelity" / scenario_name

    def matrix_dir(self, *, date: str, run_id: str) -> Path:
        return self.results_root / "reports" / date / "paper_fidelity" / f"paper_models_matrix_{run_id}"

    def matrix_manifest_path(self, *, date: str, run_id: str) -> Path:
        return self.matrix_dir(date=date, run_id=run_id) / "manifest.json"

    def matrix_failures_dir(self, *, date: str, run_id: str) -> Path:
        return self.matrix_dir(date=date, run_id=run_id) / "failures"

    def profiling_outputs_dir(self, scenario_name: str, run_id: str) -> Path:
        """Large intermediate profiling outputs (timestamped by upstream profilers)."""
        return self.tmp_root / "profiling_outputs" / scenario_name / run_id

    def profiling_root_dir(self, scenario_name: str, run_id: str) -> Path:
        """Vidur-compatible profiling root (`data/profiling/...`) for this host."""
        return self.tmp_root / "profiling_roots" / scenario_name / run_id

    def profiling_meta_path(self, scenario_name: str, run_id: str) -> Path:
        return self.profiling_root_dir(scenario_name, run_id) / "profiling_meta.json"


def build_run_meta(
    *,
    repo_root: Path,
    run_type: str,
    run_id: str,
    scenario_name: str,
    started_at: str | None = None,
    ended_at: str | None = None,
    params: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Path] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal provenance record consistent with `gpu_simulate_test.io` conventions."""
    git = get_git_info(repo_root=repo_root)
    meta: dict[str, Any] = {
        "schema_version": "v1",
        "run_type": run_type,
        "run_id": run_id,
        "scenario_name": scenario_name,
        "started_at": started_at or utcnow_iso(),
        "ended_at": ended_at,
        "git_commit": git.commit or "unknown",
        "git_dirty": git.dirty,
        "env": build_env_snapshot(),
    }
    if params is not None:
        meta["params"] = dict(params)
    if artifacts is not None:
        meta["artifacts"] = {k: str(v.resolve()) for k, v in artifacts.items()}
    if extra is not None:
        meta.update(dict(extra))
    return meta
