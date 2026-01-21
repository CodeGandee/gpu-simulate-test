"""
`vidur-cli` top-level command (argparse).

This module implements the user-facing CLI described in `specs/004-vidur-cli/`.
The CLI is intentionally implemented with Python stdlib `argparse` to minimize
dependencies and to match existing stage CLIs in this repository.

Functions
---------
build_parser
    Build the argparse parser for `vidur-cli`.
main
    Console-script entrypoint (`vidur-cli`).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from gpu_simulate_test.vidur_cli.errors import UserFacingError, format_exception_for_cli
from gpu_simulate_test.vidur_cli.resources import resolve_resources
from gpu_simulate_test.vidur_cli.search_path import discover_groups, list_presets_for_group
from gpu_simulate_test.vidur_cli.run_state import Presets, default_run_tag, normalize_run_dir, sanitize_tag
from gpu_simulate_test.vidur_cli.run_state import load_run_state
from gpu_simulate_test.vidur_cli.stages import run_init_run, run_profile, run_real, run_report, run_sim, run_trace


@dataclass(frozen=True)
class GlobalOptions:
    """Global `vidur-cli` options (apply to all subcommands)."""

    user_config: str | None
    config_dirs: list[str]
    print_resolved: bool


def build_parser() -> argparse.ArgumentParser:
    """Return the argparse parser for `vidur-cli`."""
    parser = argparse.ArgumentParser(
        prog="vidur-cli",
        description="Step-by-step Vidur sim-vs-real workflow runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  vidur-cli resources show\n"
            "  vidur-cli configs list --group model\n"
            "  vidur-cli svr init-run model=qwen3_0_6b hardware=a100 backend=transformers workload=default vidur=default\n"
            "  vidur-cli svr trace   --run-dir <run_dir> --from-lengths ./lengths.csv\n"
            "  vidur-cli svr profile --run-dir <run_dir>  # add --no-include-cpu-overhead to disable\n"
            "  vidur-cli svr sim     --run-dir <run_dir>\n"
            "  vidur-cli svr real    --run-dir <run_dir>\n"
            "  vidur-cli svr report  --run-dir <run_dir>\n"
            "\n"
            "Tip: add --print-resolved to any command to print resolved resources/config roots.\n"
        ),
    )
    parser.add_argument(
        "--user-config",
        default=None,
        help="Project-local config TOML path (default: <pwd>/.vidur-config/default.toml).",
    )
    parser.add_argument(
        "--config-dir",
        action="append",
        default=[],
        help="Additional Hydra config root (repeatable; higher precedence first).",
    )
    parser.add_argument(
        "--print-resolved",
        action="store_true",
        default=False,
        help="Print resolved resources and config roots before executing the command.",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    # resources show
    resources = sub.add_parser("resources", help="Inspect resolved resources and provenance.")
    resources_sub = resources.add_subparsers(dest="resources_cmd", required=True)
    resources_sub.add_parser("show", help="Print resolved resources and their provenance.")

    # configs list
    configs = sub.add_parser("configs", help="Inspect available Hydra preset keys.")
    configs_sub = configs.add_subparsers(dest="configs_cmd", required=True)
    configs_list = configs_sub.add_parser("list", help="List preset keys in a config group.")
    configs_list.add_argument("--group", required=True, help="Config group to list (e.g., model).")

    # sim-vs-real group (svr)
    svr = sub.add_parser("svr", help="Run the sim-vs-real workflow step-by-step.")
    svr_sub = svr.add_subparsers(dest="svr_cmd", required=True)

    init_run = svr_sub.add_parser("init-run", help="Create a run directory and initialize metadata.")
    init_run.add_argument("--run-dir", default=None, help="Run directory path (optional; relative to workspace root).")
    init_run.add_argument("--run-tag", default=None, help="Run tag override (used when allocating a new run dir).")

    trace = svr_sub.add_parser("trace", help="Materialize a canonical token-length trace under the run directory.")
    trace.add_argument("--run-dir", required=True, help="Run directory path (required; relative to workspace root).")
    trace.add_argument("--import-trace", default=None, help="Import an existing canonical trace CSV.")
    trace.add_argument("--from-lengths", default=None, help="Build a canonical trace from a lengths-only CSV.")

    profile = svr_sub.add_parser("profile", help="Run Vidur profiling and record the profiling root.")
    profile.add_argument("--run-dir", required=True, help="Run directory path (required; relative to workspace root).")
    profile.add_argument(
        "--include-cpu-overhead",
        dest="include_cpu_overhead",
        action="store_const",
        const=True,
        default=None,
        help="Override config to include CPU overhead measurement during profiling.",
    )
    profile.add_argument(
        "--no-include-cpu-overhead",
        dest="include_cpu_overhead",
        action="store_const",
        const=False,
        help="Override config to disable CPU overhead measurement during profiling.",
    )

    sim = svr_sub.add_parser("sim", help="Run Vidur simulation and record sim outputs.")
    sim.add_argument("--run-dir", required=True, help="Run directory path (required; relative to workspace root).")

    real = svr_sub.add_parser("real", help="Run real backend replay and record real outputs.")
    real.add_argument("--run-dir", required=True, help="Run directory path (required; relative to workspace root).")

    report = svr_sub.add_parser("report", help="Generate a sim-vs-real comparison report under the run directory.")
    report.add_argument("--run-dir", required=True, help="Run directory path (required; relative to workspace root).")

    return parser


def split_hydra_overrides(unknown_args: list[str]) -> list[str]:
    """Convert argparse unknown args to Hydra overrides.

    Rules (per spec):
    - Accept only `key=value` pairs
    - No `--` delimiter required
    - Reject any unknown arg without `=`
    """
    overrides: list[str] = []
    for item in unknown_args:
        if "=" not in item:
            raise UserFacingError(
                f"Unexpected argument (expected key=value override): {item!r}",
                hint="Pass Hydra-style overrides as trailing key=value arguments (e.g., model=qwen3_0_6b).",
            )
        overrides.append(item)
    return overrides


def _dispatch(*, args: argparse.Namespace, global_opts: GlobalOptions, overrides: list[str]) -> None:
    """Dispatch to a subcommand implementation."""
    resources = resolve_resources(
        pwd=Path.cwd(),
        user_config_flag=global_opts.user_config,
        cli_config_dirs=global_opts.config_dirs,
    )

    if global_opts.print_resolved:
        _print_preflight(resources)

    if args.cmd == "resources" and args.resources_cmd == "show":
        print(json.dumps(resources.to_json(), indent=2, sort_keys=True))
        return

    if args.cmd == "configs" and args.configs_cmd == "list":
        group = str(args.group)
        available_groups = discover_groups(resources.hydra_config_roots)
        if group not in available_groups:
            raise UserFacingError(
                f"Unknown config group: {group}",
                hint="Choose one of the available groups listed in the context.",
                context={"available_groups": available_groups},
                exit_code=2,
            )
        entries = list_presets_for_group(group=group, config_roots=resources.hydra_config_roots)
        for entry in entries:
            if len(entry.all_paths) > 1:
                shadowed = [str(p) for p in entry.all_paths[1:]]
                print(
                    "WARNING: preset overridden "
                    f"(group={entry.group} key={entry.key} active={entry.active_path} shadowed={shadowed})",
                    file=sys.stderr,
                )
            print(f"{entry.key}\t{entry.active_path}")
        return

    if args.cmd == "svr" and args.svr_cmd == "init-run":
        presets = _parse_required_presets(overrides)
        run_tag = sanitize_tag(str(args.run_tag)) if args.run_tag is not None else default_run_tag(presets)

        workspace_root = resources.workspace_root.value
        if args.run_dir is None:
            run_dir = (workspace_root / "sim_vs_real" / run_tag).resolve()
        else:
            run_dir = normalize_run_dir(run_dir=str(args.run_dir), workspace_root=workspace_root)

        out = run_init_run(
            run_dir=run_dir,
            run_tag=run_tag,
            presets=presets,
            overrides=overrides,
            resources=resources,
        )
        print(str(out))
        return

    if args.cmd == "svr" and args.svr_cmd == "trace":
        workspace_root = resources.workspace_root.value
        run_dir = normalize_run_dir(run_dir=str(args.run_dir), workspace_root=workspace_root)

        # Include init-run overrides for reproducibility; allow extra overrides at stage time.
        state = load_run_state(run_dir=run_dir)
        effective_overrides = list(state.get("overrides") or []) + list(overrides)

        trace_csv = run_trace(
            run_dir=run_dir,
            resources=resources,
            overrides=effective_overrides,
            import_trace=Path(args.import_trace).expanduser() if args.import_trace else None,
            from_lengths=Path(args.from_lengths).expanduser() if args.from_lengths else None,
        )
        print(str(trace_csv))
        return

    if args.cmd == "svr" and args.svr_cmd == "profile":
        workspace_root = resources.workspace_root.value
        run_dir = normalize_run_dir(run_dir=str(args.run_dir), workspace_root=workspace_root)
        state = load_run_state(run_dir=run_dir)
        effective_overrides = list(state.get("overrides") or []) + list(overrides)

        out_dir = run_profile(
            run_dir=run_dir,
            resources=resources,
            overrides=effective_overrides,
            include_cpu_overhead=args.include_cpu_overhead,
        )
        print(str(out_dir))
        return

    if args.cmd == "svr" and args.svr_cmd == "sim":
        workspace_root = resources.workspace_root.value
        run_dir = normalize_run_dir(run_dir=str(args.run_dir), workspace_root=workspace_root)
        state = load_run_state(run_dir=run_dir)
        effective_overrides = list(state.get("overrides") or []) + list(overrides)

        out_dir = run_sim(run_dir=run_dir, resources=resources, overrides=effective_overrides)
        print(str(out_dir))
        return

    if args.cmd == "svr" and args.svr_cmd == "real":
        workspace_root = resources.workspace_root.value
        run_dir = normalize_run_dir(run_dir=str(args.run_dir), workspace_root=workspace_root)
        state = load_run_state(run_dir=run_dir)
        effective_overrides = list(state.get("overrides") or []) + list(overrides)

        out_dir = run_real(run_dir=run_dir, resources=resources, overrides=effective_overrides)
        print(str(out_dir))
        return

    if args.cmd == "svr" and args.svr_cmd == "report":
        workspace_root = resources.workspace_root.value
        run_dir = normalize_run_dir(run_dir=str(args.run_dir), workspace_root=workspace_root)
        state = load_run_state(run_dir=run_dir)
        effective_overrides = list(state.get("overrides") or []) + list(overrides)

        summary_md = run_report(run_dir=run_dir, overrides=effective_overrides)
        print(str(summary_md))
        return

    raise UserFacingError(
        "This command is not implemented yet.",
        hint="Implement the remaining 004-vidur-cli tasks under specs/004-vidur-cli/tasks.md.",
        context={"cmd": getattr(args, "cmd", None)},
    )


def _print_preflight(resources) -> None:
    print("=== vidur-cli preflight: resolved resources ===", file=sys.stderr)
    print(json.dumps(resources.to_json(), indent=2, sort_keys=True), file=sys.stderr)


def _parse_required_presets(overrides: list[str]) -> Presets:
    required = ["model", "hardware", "backend", "workload", "vidur"]
    values: dict[str, str] = {}
    for item in overrides:
        key, value = item.split("=", 1)
        if key in required:
            if key in values:
                raise UserFacingError(
                    f"Duplicate preset override: {key}={value}",
                    hint="Specify each required preset exactly once: model=... hardware=... backend=... workload=... vidur=...",
                )
            values[key] = value

    missing = [k for k in required if k not in values]
    if missing:
        raise UserFacingError(
            "Missing required presets for init-run.",
            hint="Provide: model=... hardware=... backend=... workload=... vidur=...",
            context={"missing": missing, "provided": sorted(values.keys())},
        )

    return Presets(
        model=values["model"],
        hardware=values["hardware"],
        backend=values["backend"],
        workload=values["workload"],
        vidur=values["vidur"],
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `vidur-cli`."""
    parser = build_parser()
    try:
        args, unknown = parser.parse_known_args(list(argv) if argv is not None else None)
        overrides = split_hydra_overrides(list(unknown))

        global_opts = GlobalOptions(
            user_config=args.user_config,
            config_dirs=list(args.config_dir or []),
            print_resolved=bool(args.print_resolved),
        )
        _dispatch(args=args, global_opts=global_opts, overrides=overrides)
    except SystemExit:
        raise
    except BaseException as e:
        text, code = format_exception_for_cli(e)
        print(text, file=sys.stderr)
        raise SystemExit(code) from None


if __name__ == "__main__":
    main()
