# Contract: `vidur-cli` command surface

**Feature**: `specs/004-vidur-cli/spec.md`  
**Date**: 2026-01-14  

This document defines the externally visible CLI contract for `vidur-cli` v1.

## Global options

Accepted before the subcommand:

- `--user-config <path>`
  - Resolves config TOML using: `--user-config` > `GSIM_VIDUR_CLI_USER_CONFIG` > `<pwd>/.vidur-config/default.toml`.
  - Relative paths resolve relative to `pwd`.
- `--config-dir <path>` (repeatable)
  - Adds a Hydra config root with higher precedence than repo defaults.
- `--print-resolved`
  - Prints resolved repo root, workspace root, config TOML path, and Hydra config search path before executing.

Trailing `key=value` arguments are treated as Hydra overrides (no `--` delimiter required).

## Commands

### `vidur-cli resources show`

**Purpose**: Print resolved resources and their provenance.

**Exit codes**:
- `0` on success
- non-zero if a required resource cannot be resolved

**Outputs**:
- Writes nothing by default.
- With `--print-resolved`, includes config search path details.

### `vidur-cli configs list --group <group>`

**Purpose**: List available preset keys in a Hydra config group and show their source paths.

**Exit codes**:
- `0` on success
- non-zero if the group is unknown or configs cannot be loaded

### `vidur-cli svr init-run [--run-dir <path>] [--run-tag <name>] <presets...> [overrides...]`

**Purpose**: Create a run directory and initialize `run_state.json` + `resources.json`.

**Notes**:
- If `--run-dir` is omitted, a new run directory is allocated under the workspace root using the default run tag format `preset+timestamp`.
- If `--run-dir` is relative, it is interpreted relative to the workspace root.

**Exit codes**:
- `0` on success
- non-zero on invalid presets/configs or on I/O errors

### `vidur-cli svr trace --run-dir <run_dir> [...]`

**Purpose**: Materialize `trace/trace.csv` (canonical token-length trace) and `trace/trace_meta.json`.

**Exit codes**:
- `0` on success
- non-zero if prerequisites are missing or the input schema is invalid

### `vidur-cli svr profile --run-dir <run_dir> [...]`

**Purpose**: Produce a profiling root and record it in `run_state.json`.

**Exit codes**:
- `0` on success
- non-zero if prerequisites are missing or profiling fails

### `vidur-cli svr sim --run-dir <run_dir> [...]`

**Purpose**: Run Vidur simulation and record `sim_run_dir` in `run_state.json`.

**Exit codes**:
- `0` on success
- non-zero if `trace` or `profiling_root` prerequisites are missing

### `vidur-cli svr real --run-dir <run_dir> [...]`

**Purpose**: Run real backend replay and record `real_run_dir` in `run_state.json`.

**Exit codes**:
- `0` on success
- non-zero if `trace` prerequisite is missing

### `vidur-cli svr report --run-dir <run_dir> [...]`

**Purpose**: Compare one real run vs one sim run and write `report/summary.md` and artifacts.

**Exit codes**:
- `0` on success
- non-zero if `sim_run_dir` or `real_run_dir` prerequisites are missing

## Output printing rules

- On success, every command prints the primary output path (for `svr` stages, this is typically the created/used `run_dir` or stage output directory).
- On failure, commands exit non-zero and print an actionable error (including which source/prerequisite was missing).
