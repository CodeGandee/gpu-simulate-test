# Feature Specification: Vidur CLI (step-by-step sim-vs-real workflows)

**Feature Branch**: `[004-vidur-cli]`  
**Created**: 2026-01-14  
**Status**: Draft  
**Input**: User description: "we are going to implement the tool described in context/design/vidur-cli/design-of-vidur-cli.md , create 6 user stories for that, and name the new branch 004-<what>"

## Clarifications

### Session 2026-01-14

- Q: Should the CLI support selecting different default config TOMLs via a profile name (`<pwd>/.vidur-config/<profile>.toml`)? → A: No; the default is always `<pwd>/.vidur-config/default.toml`.
- Q: What should the default run directory tag format be? → A: `preset+timestamp` (include selected presets and a UTC timestamp).
- Q: Should `svr` subcommands auto-select or auto-create a run directory when `--run-dir` is omitted? → A: No; `--run-dir` is required for all `svr` subcommands except `init-run`.
- Q: How should relative config TOML paths be resolved (for `--user-config` / `GSIM_VIDUR_CLI_USER_CONFIG`)? → A: Relative paths are resolved relative to `pwd`.
- Q: How should configuration overrides be passed on the command line? → A: Unknown trailing `key=value` arguments are treated as overrides; no `--` delimiter is required.

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Preflight resource resolution (Priority: P1)

As an experiment runner, I can see exactly which repo root, model/dataset roots, workspace root, and config files/directories will be used (and where they came from) before starting a run, so I can avoid “mystery paths” and misconfigured experiments.

**Why this priority**: Clear, actionable resolution output prevents wasted time and makes the CLI safe to run from any working directory.

**Independent Test**: Can be tested by running `vidur-cli resources show` (or `--print-resolved` on any command) in an empty directory and verifying it either resolves all required paths or fails with a message listing all sources tried and the fix.

**Acceptance Scenarios**:

1. **Given** the user sets `GSIM_REPO_ROOT`, **When** they run `vidur-cli resources show`, **Then** the reported `repo_root` matches `GSIM_REPO_ROOT` and the output indicates the source was the environment.
2. **Given** the user provides `--user-config <path/to/config.toml>`, **When** they run `vidur-cli resources show`, **Then** the tool uses that config file (even if `GSIM_VIDUR_CLI_USER_CONFIG` is set) and reports the chosen config path.
3. **Given** a required resource cannot be resolved, **When** the user runs `vidur-cli resources show`, **Then** the command exits non-zero and prints which sources were attempted (env variables, project config TOML, repo fallback) and how to fix it.
4. **Given** the user provides `--user-config ./my.toml`, **When** they run `vidur-cli resources show` from directory `X`, **Then** the tool attempts to load `X/my.toml`.

---

### User Story 2 - Discover available presets and their sources (Priority: P1)

As an experiment runner, I can list the available preset keys for each config group (e.g., `model`, `hardware`, `backend`, `workload`, `vidur`) and see the source path for each preset, so I can select valid presets and understand which version “wins” when names overlap.

**Why this priority**: Users must be able to confidently choose presets and diagnose configuration overrides without reading repository internals.

**Independent Test**: Can be tested by running `vidur-cli configs list --group model` and verifying it returns a non-empty list of keys with a source path per key.

**Acceptance Scenarios**:

1. **Given** the CLI can load config directories, **When** the user runs `vidur-cli configs list --group model`, **Then** the command prints the available preset keys and the source path for each.
2. **Given** the user provides additional config directories that contain a preset key that also exists in the repo configs, **When** the user lists that group, **Then** the output indicates which preset source is active and includes a warning about the override.
3. **Given** the user requests a non-existent group, **When** they run `vidur-cli configs list --group does_not_exist`, **Then** the command exits non-zero and prints the available groups.

---

### User Story 3 - Create a run workspace (Priority: P1)

As an experiment runner, I can initialize a run directory for a sim-vs-real comparison with selected presets and a stable output location, so I can execute steps independently while keeping provenance and outputs organized.

**Why this priority**: A well-defined run directory is the foundation for step-by-step runs, reruns, and reproducibility.

**Independent Test**: Can be tested by running `vidur-cli svr init-run model=<k> hardware=<k> backend=<k> workload=<k> vidur=<k>` and verifying a run directory is created containing machine-readable metadata.

**Acceptance Scenarios**:

1. **Given** the user provides valid preset selections, **When** they run `vidur-cli svr init-run ...`, **Then** the tool creates a new run directory and prints its path.
2. **Given** the run directory is created, **When** the user inspects it, **Then** it contains `run_state.json` and `resources.json` (and an optional resolved config snapshot), recording the chosen presets and resolved resources.
3. **Given** the user passes `--run-dir <relative_path>`, **When** they run `vidur-cli svr init-run`, **Then** the run directory is created under the workspace root using that relative path.

---

### User Story 4 - Prepare a canonical token-length trace (Priority: P1)

As an experiment runner, I can materialize a canonical token-length trace dataset for a run (including arrivals), so all later stages consume the same validated input format.

**Why this priority**: Without a canonical trace, sim-vs-real runs are not comparable or reproducible.

**Independent Test**: Can be tested by creating a run directory, running `vidur-cli svr trace --run-dir <run_dir>`, and verifying it produces a valid `trace/trace.csv` and `trace/trace_meta.json`.

**Acceptance Scenarios**:

1. **Given** the user provides a valid canonical trace CSV, **When** they run `vidur-cli svr trace --run-dir <run_dir> --import-trace <path>`, **Then** the tool validates required columns and places the trace at `trace/trace.csv`.
2. **Given** the user provides a lengths-only CSV, **When** they run `vidur-cli svr trace --run-dir <run_dir> --from-lengths <path>`, **Then** the tool deterministically assigns `request_id`, generates `arrival_time_ns` according to the configured arrival schedule, and writes `trace/trace.csv`.
3. **Given** the provided CSV is missing required columns, **When** the user runs `vidur-cli svr trace ...`, **Then** the command exits non-zero and reports the missing columns and expected schema.

---

### User Story 5 - Execute profiling, simulation, and real replay as separate steps (Priority: P2)

As an experiment runner, I can run the profiling, simulation, and real replay steps separately (each reading the run directory state and recording outputs), so I can rerun only the failed or changed stages without repeating the entire workflow.

**Why this priority**: Step-by-step execution is the core usability promise of `vidur-cli`.

**Independent Test**: Can be tested by running each stage command with a prepared run directory and verifying it fails fast when prerequisites are missing and records outputs when successful.

**Acceptance Scenarios**:

1. **Given** a run directory with a prepared trace, **When** the user runs `vidur-cli svr profile --run-dir <run_dir>`, **Then** the tool records a `profiling_root` in `run_state.json` and prints the primary output path.
2. **Given** the run directory is missing `trace/trace.csv`, **When** the user runs `vidur-cli svr sim --run-dir <run_dir>`, **Then** the command exits non-zero and reports the missing prerequisite(s) without deleting any existing artifacts.
3. **Given** required prerequisites exist, **When** the user runs `vidur-cli svr sim --run-dir <run_dir>` and `vidur-cli svr real --run-dir <run_dir>`, **Then** the tool records `sim_run_dir` and `real_run_dir` in `run_state.json`.
4. **Given** any stage fails mid-run, **When** the command exits, **Then** the run directory contains a machine-readable `failure.json` describing what failed and where.
5. **Given** the user runs an `svr` stage command other than `init-run` without `--run-dir`, **When** the command starts, **Then** it exits non-zero and tells the user to provide `--run-dir` (for example, using the path printed by `svr init-run`).

---

### User Story 6 - Generate a sim-vs-real comparison report (Priority: P2)

As an experiment runner, I can generate a comparison report for one sim run vs one real run, so I can quickly assess timing fidelity and key caveats (arrival kind, CPU overhead settings) without manual plotting.

**Why this priority**: The workflow’s end product is a clear summary that supports decision-making and debugging.

**Independent Test**: Can be tested by running `vidur-cli svr report --run-dir <run_dir>` after `sim_run_dir` and `real_run_dir` are recorded, and verifying it produces a `summary.md` and report artifacts.

**Acceptance Scenarios**:

1. **Given** both `sim_run_dir` and `real_run_dir` are recorded in `run_state.json`, **When** the user runs `vidur-cli svr report --run-dir <run_dir>`, **Then** the tool writes `summary.md` and report artifacts under the run directory and prints the report path.
2. **Given** either `sim_run_dir` or `real_run_dir` is missing, **When** the user runs `vidur-cli svr report --run-dir <run_dir>`, **Then** the command exits non-zero and reports which prerequisite is missing.
3. **Given** CPU overhead was disabled for the run, **When** the user generates a report, **Then** `summary.md` includes a clear warning about CPU overhead being disabled.

### Edge Cases

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right edge cases.
-->

- What happens when the config TOML path is explicitly provided but does not exist?
- What happens when the default config path `<pwd>/.vidur-config/default.toml` does not exist?
- How does the CLI behave when the same preset key exists in multiple config directories (user + repo)?
- How does the CLI behave when the run directory exists but has incomplete or corrupted `run_state.json`?
- How does the CLI handle a trace CSV with wrong column types or missing required columns?
- How does the CLI handle a `--run-dir` relative path when the workspace root is also relative?
- What happens when the workspace root is not writable?
- What happens when a stage partially succeeds (some outputs written) and then fails?

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: The CLI MUST provide a `vidur-cli` command with subcommand groups `resources`, `configs`, and `sim-vs-real` (with `svr` as an alias for `sim-vs-real`).
- **FR-002**: The CLI MUST support global options `--user-config <path>`, `--config-dir <path>` (repeatable), and `--print-resolved`.
- **FR-003**: The CLI MUST resolve the config TOML path using: `--user-config` > `GSIM_VIDUR_CLI_USER_CONFIG` > `<pwd>/.vidur-config/default.toml` (no implicit “profile name” selection in v1).
- **FR-004**: If the user explicitly provides a config TOML path (via `--user-config` or `GSIM_VIDUR_CLI_USER_CONFIG`) and it does not exist, the CLI MUST exit non-zero with an actionable error.
- **FR-005**: If the default config TOML path `<pwd>/.vidur-config/default.toml` does not exist, the CLI MUST continue using environment variables and repo fallbacks (no hard failure).
- **FR-006**: The CLI MUST resolve `repo_root`, `models_root`, `datasets_root`, and `workspace_dir` with precedence: environment variables > project config TOML > repo fallback.
- **FR-007**: When a resource cannot be resolved, the CLI MUST exit non-zero and the error message MUST list each source attempted and the next step to fix it.
- **FR-008**: When `--print-resolved` is provided, the CLI MUST print the resolved resources and config search path and indicate the chosen source for each resolved value.
- **FR-009**: The CLI MUST support configuration directories from (highest precedence to lowest): `--config-dir` (in order) > `GSIM_VIDUR_CLI_HYDRA_CONFIG_DIRS` > config TOML `hydra.config_dirs` > repo default config root.
- **FR-010**: The CLI MUST warn when a user-provided config directory overrides a preset key that also exists in the repo configs, and the warning MUST include both source paths.
- **FR-011**: `vidur-cli configs list --group <group>` MUST list available preset keys for the requested group and indicate the source path for each preset.
- **FR-012**: `vidur-cli resources show` MUST print the resolved resource map (including `repo_root`, `models_root`, `datasets_root`, and `workspace_root`).
- **FR-013**: `vidur-cli` MUST accept configuration overrides as trailing `key=value` arguments (unknown args treated as overrides; `--` delimiter is not required) and apply them consistently across subcommands.
- **FR-014**: `svr init-run` MUST create (or select) a run directory and write `run_state.json` and `resources.json`, recording preset selections and resolved resources, and print the run directory path.
- **FR-015**: When `--run-dir` is provided to any `svr` subcommand, a relative path MUST be interpreted relative to the workspace root.
- **FR-016**: For all `svr` subcommands except `init-run`, `--run-dir` MUST be required; when omitted, the CLI MUST exit non-zero with an actionable message.
- **FR-017**: `svr trace` MUST produce `trace/trace.csv` and `trace/trace_meta.json` under the run directory, and the produced CSV MUST include required columns `request_id`, `arrival_time_ns`, `num_prefill_tokens`, and `num_decode_tokens`.
- **FR-018**: `svr trace` MUST support importing an existing canonical trace CSV and MUST fail with a clear schema validation error when required columns are missing.
- **FR-019**: `svr trace` MUST support building a canonical trace from a lengths-only CSV by deterministically assigning `request_id` and deterministically generating arrivals from a configured arrival schedule; trace metadata MUST record inputs and determinism controls (e.g., seed).
- **FR-020**: `svr profile` MUST record `profiling_root` in `run_state.json` on success and MUST include CPU overhead measurement by default, with an explicit option to disable it.
- **FR-021**: `svr sim` MUST require both a canonical trace and a profiling root; if either is missing, it MUST exit non-zero with a message listing missing prerequisites.
- **FR-022**: `svr sim` MUST record `sim_run_dir` in `run_state.json` on success.
- **FR-023**: `svr real` MUST require a canonical trace; if missing, it MUST exit non-zero with a message listing missing prerequisites.
- **FR-024**: `svr real` MUST record `real_run_dir` in `run_state.json` on success.
- **FR-025**: `svr report` MUST require `sim_run_dir` and `real_run_dir`; if either is missing, it MUST exit non-zero and report which is missing.
- **FR-026**: `svr report` MUST write `summary.md` and report artifacts under the run directory, and `summary.md` MUST state the workload arrival kind and CPU overhead status, including a warning if CPU overhead was disabled.
- **FR-027**: Each `svr` stage command MUST write machine-readable failure metadata (e.g., `failure.json`) on failure and MUST NOT delete partial outputs.
- **FR-028**: By default, outputs MUST be written under the resolved workspace root; writing to repo `tmp/` or `results/` MUST only happen if the user explicitly configures the workspace root to point there.
- **FR-029**: All successful commands MUST print the primary output path(s), and all failures MUST exit with a non-zero code.
- **FR-030**: By default, `svr init-run` MUST generate a filesystem-safe run tag that includes the selected preset keys (`model`, `hardware`, `backend`, `workload`, `vidur`) and a UTC timestamp.
- **FR-031**: Relative paths provided via `--user-config` or `GSIM_VIDUR_CLI_USER_CONFIG` MUST be resolved relative to `pwd`.

### Assumptions

- Users run `vidur-cli` in an environment where the Vidur simulator and at least one real backend runner are available.
- Users have access to required model/dataset files (either via repository defaults or configured paths).

### Dependencies

- Vidur simulator is available for profiling and simulation stages.
- A real backend runner is available for the “real replay” stage.

### Out of Scope

- Aggregating results across many runs into a single combined report.
- A sweep command for multi-case execution and aggregation.
- Remote execution or cluster orchestration.
- Any graphical user interface.

### Key Entities *(include if feature involves data)*

- **Config Sources**: The set of inputs that define presets and overrides (config TOML path, config directories, and override arguments).
- **Resource Map**: Resolved roots and paths used by the workflow (`repo_root`, `models_root`, `datasets_root`, `workspace_root`, and per-asset mappings).
- **Run Directory**: A workspace folder representing one sim-vs-real run, identified by a run tag and containing all artifacts and metadata.
- **Run State**: A machine-readable record of selected presets, resolved resources, timestamps, stage status, and produced artifact paths.
- **Trace Dataset**: A canonical token-length trace CSV plus metadata describing its source and arrival schedule determinism controls.
- **Profiling Artifact**: A profiling output root produced by the profiling stage and reused by simulation.
- **Simulation Run**: The simulation output directory (or reference) produced by the sim stage.
- **Real Run**: The real replay output directory (or reference) produced by the real stage.
- **Report**: The comparison summary and artifacts produced from one simulation run and one real run.

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: Users can create a new run directory with `svr init-run` in a single command and the run directory contains `run_state.json` and `resources.json`.
- **SC-002**: For any failed resource resolution, the CLI error output lists all sources attempted and includes at least one actionable fix step.
- **SC-003**: When trace generation is configured with the same inputs and determinism controls, `svr trace` produces identical `trace/trace.csv` outputs across repeated runs.
- **SC-004**: After each successful stage command (`trace`, `profile`, `sim`, `real`, `report`), `run_state.json` contains the stage’s output reference(s) and a timestamp.
- **SC-005**: Users can complete a full sim-vs-real comparison (init-run → trace → profile → sim → real → report) using no more than 6 commands, each printing the primary output path.
