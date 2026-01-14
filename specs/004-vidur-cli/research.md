# Research: Vidur CLI design decisions (Phase 0)

**Feature**: `specs/004-vidur-cli/spec.md`  
**Date**: 2026-01-14  

This document captures key implementation decisions needed to implement `vidur-cli` as specified in `spec.md` and `context/design/vidur-cli/design-of-vidur-cli.md`.

## Decisions

### 1) CLI framework

- **Decision**: Use Python stdlib `argparse` for the top-level `vidur-cli` command and subcommands.
- **Rationale**: Minimizes dependencies, matches existing “python -m …” CLIs, and is sufficient for nested subcommands + global flags + pass-through `key=value` overrides.
- **Alternatives considered**:
  - `click`/`typer`: nicer UX but adds dependency surface and rework risk.

### 2) Config TOML location (project-local)

- **Decision**: Default config TOML path is always `<pwd>/.vidur-config/default.toml` (no profile mechanism).
- **Rationale**: Keeps configuration local to a working directory (portable, shareable with experiments) and matches recorded clarifications in the spec.
- **Alternatives considered**:
  - Profile-based defaults (`<pwd>/.vidur-config/<profile>.toml`): adds naming UX and more states to support.
  - Home-directory defaults: explicitly rejected by the design constraints.

### 3) Relative path resolution for config TOML

- **Decision**: Relative paths provided via `--user-config` or `GSIM_VIDUR_CLI_USER_CONFIG` are resolved relative to `pwd`.
- **Rationale**: Consistent with “project-local” config; avoids surprising coupling to the resolved `repo_root`.
- **Alternatives considered**:
  - Resolve relative to `repo_root`: can be surprising when running from outside the repo.
  - Force absolute-only: unnecessary friction.

### 4) Hydra integration approach

- **Decision**: Use Hydra’s programmatic composition (`initialize`/`compose`) to load stage configs from `configs/compare_vidur_real`, add extra config roots, and apply trailing `key=value` overrides.
- **Rationale**: Preserves Hydra-native group semantics (`model=...`, etc.) while allowing `vidur-cli` to control output directories and metadata writing (instead of relying on Hydra’s `hydra.run.dir` and `job.chdir`).
- **Alternatives considered**:
  - Invoke existing `@hydra.main` stage CLIs directly: conflicts with `svr` run-dir ownership (Hydra will chdir and write under repo `tmp/` unless heavily overridden).
  - Hand-merge YAMLs with OmegaConf: loses Hydra group composition and search path behavior.

### 5) Config search path precedence

- **Decision**: Build the Hydra search path as:
  1) `--config-dir` entries (in provided order)
  2) `GSIM_VIDUR_CLI_HYDRA_CONFIG_DIRS` entries (split by OS path separator)
  3) `hydra.config_dirs` from the resolved config TOML
  4) Repo default config root: `<repo_root>/configs/compare_vidur_real`
- **Rationale**: Matches the design doc (user config has highest priority), while ensuring `vidur-cli` has a consistent repo fallback.
- **Alternatives considered**:
  - Include all repo `configs/`: explicitly rejected to keep v1 end-user surface focused.

### 6) Run directory selection rules

- **Decision**:
  - `svr init-run` may allocate a new run directory when `--run-dir` is omitted.
  - All other `svr` subcommands require `--run-dir` (no auto-selection).
- **Rationale**: Prevents accidental writes to the wrong run when multiple runs exist; aligns with the spec clarifications.
- **Alternatives considered**:
  - “Most recent run” selection: convenient but easy to misuse in multi-run workspaces.
  - Implicit parent-dir detection: can behave unexpectedly when invoked from nested paths.

### 7) Default run tag format

- **Decision**: Default run tag is `preset+timestamp` (include `model/hardware/backend/workload/vidur` keys + UTC timestamp), sanitized for filesystem safety.
- **Rationale**: Human-readable and collision-resistant.
- **Alternatives considered**:
  - Timestamp-only: loses context.
  - Random slug: loses debuggability.

### 8) Output location policy

- **Decision**: All `vidur-cli` outputs are rooted under the resolved workspace root; repo `tmp/`/`results/` are used only if the user explicitly points the workspace root there.
- **Rationale**: Enables running from anywhere and avoids polluting the repo.
- **Alternatives considered**:
  - Always write to repo `tmp/`: breaks portability and is fragile in shared repos.

### 9) Artifact contracts and schema versioning

- **Decision**: Persist machine-readable artifacts with `schema_version: "v1"` and stable, explicit fields:
  - `run_state.json` as the canonical state machine for a run
  - `resources.json` recording resolved resource values and their sources
  - `failure.json` written on stage failure
  - `trace/trace.csv` + `trace/trace_meta.json` for canonical trace inputs
- **Rationale**: Supports reproducibility and future migration without breaking older runs.
- **Alternatives considered**:
  - “Implicit by convention” artifacts: harder to validate and harder to support long-term.

### 10) Canonical trace schema choice (sim-vs-real)

- **Decision**: Canonical trace CSV schema is:
  - Required: `request_id`, `arrival_time_ns`, `num_prefill_tokens`, `num_decode_tokens`
  - Optional: any extra columns preserved for provenance
- **Rationale**: Aligns with existing compare workflow timing units (`arrival_time_ns`) and replay logic.
- **Alternatives considered**:
  - Seconds-based `arrived_at`: used in paper-fidelity modules; not the v1 target for this CLI.
