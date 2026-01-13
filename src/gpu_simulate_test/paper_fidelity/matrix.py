from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Literal

from gpu_simulate_test.paper_fidelity.failure_record import (
    build_failure_record,
    categorize_blocker,
    write_failure_record,
)
from gpu_simulate_test.paper_fidelity.matrix_manifest import MatrixRunEntry, write_matrix_manifest
from gpu_simulate_test.paper_fidelity.paper_models import PAPER_MODEL_SCENARIOS, validate_paper_model_scenarios
from gpu_simulate_test.paper_fidelity.paths import PaperFidelityPaths

Workload = Literal["static", "dynamic"]
Scale = Literal["small", "medium", "full"]


@dataclass(frozen=True)
class MatrixArgs:
    run_id: str
    scenarios: list[str]
    workloads: list[Workload]
    scale: Scale
    include_cpu_overhead: bool
    stop_on_failure: bool


def default_matrix_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def _utc_date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _file_safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()) or "unknown"


def _last_non_empty_line(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line:
            return line
    return None


def _run_subprocess(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def run_matrix(*, repo_root: Path, args: MatrixArgs) -> Path:
    """Run the profile+repro matrix and return the manifest.json path."""
    scenarios = args.scenarios or list(PAPER_MODEL_SCENARIOS)
    validate_paper_model_scenarios(scenarios)

    pf_paths = PaperFidelityPaths(repo_root=repo_root)
    date = _utc_date_str()
    manifest_path = pf_paths.matrix_manifest_path(date=date, run_id=args.run_id)
    failures_dir = pf_paths.matrix_failures_dir(date=date, run_id=args.run_id)
    failures_dir.mkdir(parents=True, exist_ok=True)

    scenario_tag_suffix = _file_safe_slug(args.run_id)

    runs: list[MatrixRunEntry] = []
    for scenario_key in scenarios:
        scenario_tag = f"{scenario_key}_matrix_{scenario_tag_suffix}"

        profile_cmd = [
            sys.executable,
            "-m",
            "gpu_simulate_test.cli.paper_fidelity",
            "profile",
            "--scenario",
            scenario_key,
        ]
        if args.include_cpu_overhead:
            profile_cmd.append("--include-cpu-overhead")
        profile_cmd.append("scenario.name=" + scenario_tag)

        profile_proc = _run_subprocess(profile_cmd, cwd=repo_root)
        if profile_proc.returncode != 0:
            stdout_last = _last_non_empty_line(profile_proc.stdout) or ""
            stderr_last = _last_non_empty_line(profile_proc.stderr) or ""
            error_message = stderr_last or stdout_last or f"profile failed (returncode={profile_proc.returncode})"
            traceback_text = profile_proc.stderr.strip() or None

            category = categorize_blocker(error_message=error_message, traceback=traceback_text)
            failure_record = build_failure_record(
                run_id=args.run_id,
                action="profile",
                scenario_key=scenario_key,
                scenario_name=scenario_tag,
                workload=None,
                scale=None,
                attempted_command=profile_cmd,
                hydra_overrides=[f"scenario.name={scenario_tag}"],
                error_message=error_message,
                traceback=traceback_text,
                blocker_category=category,
            )
            failure_path = write_failure_record(
                failures_dir / f"{_file_safe_slug(args.run_id)}_{_file_safe_slug(scenario_key)}_profile.json",
                failure_record,
            )

            for workload in args.workloads:
                runs.append(
                    MatrixRunEntry(
                        scenario_key=scenario_key,
                        workload=workload,
                        scale=args.scale,
                        status="failure",
                        report_dir=None,
                        failure_record_json=failure_path,
                        blocker_category=category,
                    )
                )
            if args.stop_on_failure:
                break
            continue

        profiling_root_line = _last_non_empty_line(profile_proc.stdout)
        if not profiling_root_line:
            raise RuntimeError(f"profile produced no output for scenario={scenario_key}")

        profiling_root = Path(profiling_root_line).expanduser()
        if not profiling_root.is_absolute():
            profiling_root = (repo_root / profiling_root).resolve()
        else:
            profiling_root = profiling_root.resolve()

        for workload in args.workloads:
            report_name = scenario_tag if workload == "static" else f"{scenario_tag}_{workload}_{args.scale}"
            repro_cmd = [
                sys.executable,
                "-m",
                "gpu_simulate_test.cli.paper_fidelity",
                "repro",
                "--scenario",
                scenario_key,
                "--workload",
                workload,
                "--scale",
                args.scale,
                "scenario.name=" + scenario_tag,
                "scenario.vidur.profiling_root=" + str(profiling_root),
            ]
            repro_proc = _run_subprocess(repro_cmd, cwd=repo_root)
            if repro_proc.returncode != 0:
                stdout_last = _last_non_empty_line(repro_proc.stdout) or ""
                stderr_last = _last_non_empty_line(repro_proc.stderr) or ""
                error_message = stderr_last or stdout_last or f"repro failed (returncode={repro_proc.returncode})"
                traceback_text = repro_proc.stderr.strip() or None

                category = categorize_blocker(error_message=error_message, traceback=traceback_text)
                failure_record = build_failure_record(
                    run_id=args.run_id,
                    action="repro",
                    scenario_key=scenario_key,
                    scenario_name=report_name,
                    workload=workload,
                    scale=args.scale,
                    attempted_command=repro_cmd,
                    hydra_overrides=[
                        f"scenario.name={scenario_tag}",
                        f"scenario.vidur.profiling_root={profiling_root}",
                    ],
                    error_message=error_message,
                    traceback=traceback_text,
                    blocker_category=category,
                )
                failure_path = write_failure_record(
                    failures_dir
                    / f"{_file_safe_slug(args.run_id)}_{_file_safe_slug(scenario_key)}_repro_{workload}.json",
                    failure_record,
                )
                runs.append(
                    MatrixRunEntry(
                        scenario_key=scenario_key,
                        workload=workload,
                        scale=args.scale,
                        status="failure",
                        report_dir=None,
                        failure_record_json=failure_path,
                        blocker_category=category,
                    )
                )
                if args.stop_on_failure:
                    break
                continue

            report_dir_line = _last_non_empty_line(repro_proc.stdout)
            if not report_dir_line:
                raise RuntimeError(f"repro produced no output for scenario={scenario_key} workload={workload}")
            report_dir = Path(report_dir_line).expanduser()
            if not report_dir.is_absolute():
                report_dir = (repo_root / report_dir).resolve()
            else:
                report_dir = report_dir.resolve()

            runs.append(
                MatrixRunEntry(
                    scenario_key=scenario_key,
                    workload=workload,
                    scale=args.scale,
                    status="success",
                    report_dir=report_dir,
                    failure_record_json=None,
                    blocker_category=None,
                )
            )

        if args.stop_on_failure and any(r.status == "failure" for r in runs if r.scenario_key == scenario_key):
            break

    manifest = write_matrix_manifest(
        out_path=manifest_path,
        repo_root=repo_root,
        run_id=args.run_id,
        scenarios=list(scenarios),
        workloads=[str(w) for w in args.workloads],
        scale=str(args.scale),
        runs=runs,
    )
    return manifest
