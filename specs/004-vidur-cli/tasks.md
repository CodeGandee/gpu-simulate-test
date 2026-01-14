---
description: "Task list for implementing vidur-cli"
---

# Tasks: Vidur CLI (step-by-step sim-vs-real workflows)

**Input**: Design documents from `specs/004-vidur-cli/`  
**Prerequisites**: `specs/004-vidur-cli/plan.md`, `specs/004-vidur-cli/spec.md`, `specs/004-vidur-cli/research.md`, `specs/004-vidur-cli/data-model.md`, `specs/004-vidur-cli/contracts/`, `specs/004-vidur-cli/quickstart.md`

**Tests**: No automated tests are explicitly requested in `specs/004-vidur-cli/spec.md`. Tasks focus on CLI-level manual verification and artifact/schema correctness.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `- [ ] T### [P?] [US#?] Description with file path`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[US#]**: Which user story this task belongs to (e.g., `[US1]`)
- **File paths are required** in every task description

## Path Conventions

- Single project: `src/`, `tests/` at repository root
- CLI entrypoints: `src/gpu_simulate_test/cli/`
- Shared `vidur-cli` helpers: `src/gpu_simulate_test/vidur_cli/`
- Feature docs live under `specs/004-vidur-cli/`

## Implementation Guides

- `context/tasks/working/004-vidur-cli/impl-phase-1-setup.md`
- `context/tasks/working/004-vidur-cli/impl-phase-2-foundational.md`
- `context/tasks/working/004-vidur-cli/impl-phase-3-us1-resources.md`
- `context/tasks/working/004-vidur-cli/impl-phase-4-us2-configs.md`
- `context/tasks/working/004-vidur-cli/impl-phase-5-us3-run-workspace.md`
- `context/tasks/working/004-vidur-cli/impl-phase-6-us4-trace.md`
- `context/tasks/working/004-vidur-cli/impl-phase-7-us5-stages.md`
- `context/tasks/working/004-vidur-cli/impl-phase-8-us6-report.md`
- `context/tasks/working/004-vidur-cli/impl-phase-9-polish.md`
- `context/tasks/working/004-vidur-cli/impl-integrate-phases.md`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project wiring and feature scaffolding

- [ ] T001 Initialize required submodules under `extern/tracked/vidur/` and `extern/tracked/sarathi-serve/` (see `.gitmodules`)
- [ ] T002 Materialize the dev environment with `pixi install` (inputs: `pyproject.toml`, `pixi.lock`)
- [ ] T003 Add a `vidur-cli` console script and a Pixi task entry in `pyproject.toml`
- [ ] T004 Create the CLI entrypoint module skeleton in `src/gpu_simulate_test/cli/vidur_cli.py`
- [ ] T005 Create the helper package initializer in `src/gpu_simulate_test/vidur_cli/__init__.py`
- [ ] T006 [P] Add error/exception scaffolding in `src/gpu_simulate_test/vidur_cli/errors.py`
- [ ] T007 [P] Add resource resolution scaffolding in `src/gpu_simulate_test/vidur_cli/resources.py`
- [ ] T008 [P] Add Hydra search-path scaffolding in `src/gpu_simulate_test/vidur_cli/search_path.py`
- [ ] T009 [P] Add run-state scaffolding in `src/gpu_simulate_test/vidur_cli/run_state.py`
- [ ] T010 [P] Add trace generation scaffolding in `src/gpu_simulate_test/vidur_cli/trace.py`
- [ ] T011 [P] Add stage runner scaffolding in `src/gpu_simulate_test/vidur_cli/stages.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T012 Implement the `argparse` CLI skeleton + subcommand dispatch in `src/gpu_simulate_test/cli/vidur_cli.py`
- [ ] T013 Implement trailing `key=value` override parsing (`parse_known_args`) in `src/gpu_simulate_test/cli/vidur_cli.py`
- [ ] T014 Implement top-level error handling (exit codes, stderr formatting) in `src/gpu_simulate_test/vidur_cli/errors.py`
- [ ] T015 Implement project-local config TOML parsing (stdlib `tomllib`) in `src/gpu_simulate_test/vidur_cli/resources.py`
- [ ] T016 Implement `resources.json` write helper (schema v1, absolute paths) in `src/gpu_simulate_test/vidur_cli/resources.py`
- [ ] T017 Implement Hydra config root resolution (precedence + normalization) in `src/gpu_simulate_test/vidur_cli/search_path.py`
- [ ] T018 Implement config filesystem scanning helpers (groups + preset keys) in `src/gpu_simulate_test/vidur_cli/search_path.py`
- [ ] T019 Implement Hydra programmatic composition helper (initialize/compose using resolved config roots) in `src/gpu_simulate_test/vidur_cli/search_path.py`
- [ ] T020 Implement `run_state.json` read/write helpers (schema v1, timestamps) in `src/gpu_simulate_test/vidur_cli/run_state.py`
- [ ] T021 Implement `failure.json` write helper + stage wrapper for exceptions in `src/gpu_simulate_test/vidur_cli/run_state.py`
- [ ] T022 Implement run directory normalization + prerequisite helpers in `src/gpu_simulate_test/vidur_cli/run_state.py`

**Checkpoint**: Foundation ready — user story implementation can now begin.

---

## Phase 3: User Story 1 - Preflight resource resolution (Priority: P1) 🎯 MVP

**Goal**: Resolve repo/workspace roots + config paths with clear provenance and safe failure modes.

**Independent Test**: Run `pixi run vidur-cli resources show` in an empty directory and verify it either resolves all required resources or fails non-zero with an actionable message listing sources tried.

### Implementation for User Story 1

- [ ] T023 [US1] Implement config TOML path resolution (`--user-config` > `GSIM_VIDUR_CLI_USER_CONFIG` > `<pwd>/.vidur-config/default.toml`) in `src/gpu_simulate_test/vidur_cli/resources.py`
- [ ] T024 [US1] Implement `repo_root` resolution + validation (`GSIM_REPO_ROOT` > `resources.repo_root` > `pwd`) in `src/gpu_simulate_test/vidur_cli/resources.py`
- [ ] T025 [P] [US1] Implement `models_root` and `datasets_root` resolution (`GSIM_MODELS_ROOT`/`GSIM_DATASETS_ROOT` > TOML > `<repo_root>/models|datasets`) in `src/gpu_simulate_test/vidur_cli/resources.py`
- [ ] T026 [P] [US1] Implement `workspace_root` resolution from `GSIM_VIDUR_WORKSPACE_DIR` / `resources.workspace_dir` (relative ⇒ `<pwd>/.vidur-output/<dir>`) in `src/gpu_simulate_test/vidur_cli/resources.py`
- [ ] T027 [US1] Implement `vidur-cli resources show` handler and output formatting in `src/gpu_simulate_test/cli/vidur_cli.py`
- [ ] T028 [US1] Implement `--print-resolved` preflight printing (resources + config roots) in `src/gpu_simulate_test/cli/vidur_cli.py`
- [ ] T029 [US1] Ensure resource-resolution failures list attempted sources + fix steps in `src/gpu_simulate_test/vidur_cli/errors.py`
- [ ] T030 [US1] Document env vars + TOML keys for resolution in `specs/004-vidur-cli/quickstart.md`

**Checkpoint**: `resources show` and `--print-resolved` are usable from any working directory.

---

## Phase 4: User Story 2 - Discover available presets and their sources (Priority: P1)

**Goal**: List available Hydra preset keys per group and show which config root provides each.

**Independent Test**: Run `pixi run vidur-cli configs list --group model` and verify it prints a non-empty list of keys with a source path per key.

### Implementation for User Story 2

- [ ] T031 [US2] Implement available-group discovery across resolved config roots in `src/gpu_simulate_test/vidur_cli/search_path.py`
- [ ] T032 [US2] Implement preset key listing for `--group <group>` (key → active source path) in `src/gpu_simulate_test/vidur_cli/search_path.py`
- [ ] T033 [US2] Emit override warnings when a higher-precedence config root shadows a lower-precedence preset in `src/gpu_simulate_test/vidur_cli/search_path.py`
- [ ] T034 [US2] Implement `vidur-cli configs list --group <group>` handler + printing in `src/gpu_simulate_test/cli/vidur_cli.py`
- [ ] T035 [US2] Implement unknown-group error (exit non-zero, list available groups) in `src/gpu_simulate_test/cli/vidur_cli.py`
- [ ] T036 [US2] Add `configs list` examples (including override warning expectations) to `specs/004-vidur-cli/quickstart.md`

**Checkpoint**: Users can discover valid preset keys without reading repo internals.

---

## Phase 5: User Story 3 - Create a run workspace (Priority: P1)

**Goal**: Initialize a run directory with selected presets and machine-readable provenance.

**Independent Test**: Run `pixi run vidur-cli svr init-run model=<k> hardware=<k> backend=<k> workload=<k> vidur=<k>` and verify the printed run directory contains `run_state.json` and `resources.json`.

### Implementation for User Story 3

- [ ] T037 [US3] Implement filesystem-safe run tag generation (`preset+timestamp`, UTC) in `src/gpu_simulate_test/vidur_cli/run_state.py`
- [ ] T038 [US3] Validate `svr init-run` includes required preset overrides (`model=... hardware=... backend=... workload=... vidur=...`) in `src/gpu_simulate_test/cli/vidur_cli.py`
- [ ] T039 [US3] Implement run-dir selection rules (`--run-dir` optional; relative ⇒ under `workspace_root`; default ⇒ `<workspace_root>/sim_vs_real/<run_tag>`) in `src/gpu_simulate_test/cli/vidur_cli.py`
- [ ] T040 [US3] Implement `svr init-run` stage runner (mkdir run dir, write `run_state.json`) in `src/gpu_simulate_test/vidur_cli/stages.py`
- [ ] T041 [US3] Write `resources.json` snapshot during init-run in `src/gpu_simulate_test/vidur_cli/stages.py`
- [ ] T042 [US3] Write optional resolved config snapshot to `<run_dir>/resolved_config.yaml` in `src/gpu_simulate_test/vidur_cli/stages.py`
- [ ] T043 [US3] Wire `svr init-run` CLI to the stage runner and print the run dir path in `src/gpu_simulate_test/cli/vidur_cli.py`
- [ ] T044 [US3] Ensure init-run failures write `failure.json` when possible (stage=`init-run`) in `src/gpu_simulate_test/vidur_cli/stages.py`

**Checkpoint**: A run directory can be created once and reused by subsequent stages.

---

## Phase 6: User Story 4 - Prepare a canonical token-length trace (Priority: P1)

**Goal**: Materialize a validated canonical `trace/trace.csv` + `trace/trace_meta.json` for a run.

**Independent Test**: Run `pixi run vidur-cli svr trace --run-dir <run_dir>` and verify it produces `trace/trace.csv` and `trace/trace_meta.json`.

### Implementation for User Story 4

- [ ] T045 [US4] Implement canonical trace CSV validation (required columns + monotonic `arrival_time_ns`) in `src/gpu_simulate_test/vidur_cli/trace.py`
- [ ] T046 [US4] Implement `trace_meta.json` writing per `specs/004-vidur-cli/contracts/trace_meta.schema.json` in `src/gpu_simulate_test/vidur_cli/trace.py`
- [ ] T047 [US4] Implement `--import-trace <path>` flow (validate + copy to `<run_dir>/trace/trace.csv`) in `src/gpu_simulate_test/vidur_cli/trace.py`
- [ ] T048 [US4] Implement `--from-lengths <path>` flow (deterministic `request_id`, deterministic arrivals) in `src/gpu_simulate_test/vidur_cli/trace.py`
- [ ] T049 [US4] Implement default trace generation (no flags) from workload config prompts/tokenizer + arrival schedule in `src/gpu_simulate_test/vidur_cli/trace.py`
- [ ] T050 [P] [US4] Write compatibility artifacts `trace/trace_lengths.csv` and `trace/trace_intervals.csv` in `src/gpu_simulate_test/vidur_cli/trace.py`
- [ ] T051 [US4] Implement `svr trace` stage runner (write artifacts, update `run_state.json`, write `failure.json` on errors) in `src/gpu_simulate_test/vidur_cli/stages.py`
- [ ] T052 [US4] Wire `svr trace` flags (`--import-trace`, `--from-lengths`) + `--run-dir` requirement in `src/gpu_simulate_test/cli/vidur_cli.py`
- [ ] T053 [US4] Update trace examples and flags in `specs/004-vidur-cli/quickstart.md`

**Checkpoint**: A canonical trace exists and later stages can depend on it.

---

## Phase 7: User Story 5 - Execute profiling, simulation, and real replay as separate steps (Priority: P2)

**Goal**: Run `profile`, `sim`, and `real` independently; each stage records outputs and failures in the run directory.

**Independent Test**: Run each stage with `--run-dir <run_dir>` and verify it fails fast when prerequisites are missing and records stage outputs in `run_state.json` when successful.

### Implementation for User Story 5

- [ ] T054 [US5] Add `svr profile|sim|real` subcommands (all require `--run-dir`; profile includes `--include-cpu-overhead/--no-include-cpu-overhead`) in `src/gpu_simulate_test/cli/vidur_cli.py`
- [ ] T055 [US5] Implement prerequisite checks (trace exists; profiling_root exists for sim; etc.) in `src/gpu_simulate_test/vidur_cli/run_state.py`
- [ ] T056 [US5] Implement `svr profile` stage runner (outputs under `<run_dir>/profile`, updates `run_state.json`) in `src/gpu_simulate_test/vidur_cli/stages.py`
- [ ] T057 [US5] Implement `svr sim` stage runner (outputs under `<run_dir>/sim`, requires trace + profiling root, updates `run_state.json`) in `src/gpu_simulate_test/vidur_cli/stages.py`
- [ ] T058 [P] [US5] Implement token-length replay helpers for backends (Transformers/Sarathi prompt_token_ids) in `src/gpu_simulate_test/vidur_cli/real_runner.py`
- [ ] T059 [US5] Implement `svr real` stage runner (outputs under `<run_dir>/real`, writes `request_metrics.csv`/`token_metrics.csv`, updates `run_state.json`) in `src/gpu_simulate_test/vidur_cli/stages.py`
- [ ] T060 [US5] Ensure `profile|sim|real` write `failure.json` on exceptions and preserve partial outputs in `src/gpu_simulate_test/vidur_cli/stages.py`
- [ ] T061 [US5] Ensure each stage prints its primary output path and exits non-zero on failure in `src/gpu_simulate_test/cli/vidur_cli.py`
- [ ] T062 [US5] Update stage workflow examples (and `--run-dir` requirement) in `specs/004-vidur-cli/quickstart.md`

**Checkpoint**: `profile`, `sim`, and `real` are independently runnable and resumable.

---

## Phase 8: User Story 6 - Generate a sim-vs-real comparison report (Priority: P2)

**Goal**: Generate `report/summary.md` and artifacts for one sim run vs one real run.

**Independent Test**: After `sim_run_dir` and `real_run_dir` are recorded, run `pixi run vidur-cli svr report --run-dir <run_dir>` and verify it writes `report/summary.md` and prints the report path.

### Implementation for User Story 6

- [ ] T063 [US6] Add `svr report` subcommand (requires `--run-dir`) in `src/gpu_simulate_test/cli/vidur_cli.py`
- [ ] T064 [US6] Implement report prerequisites (require `sim_run_dir` + `real_run_dir`) in `src/gpu_simulate_test/vidur_cli/run_state.py`
- [ ] T065 [US6] Implement report generation under `<run_dir>/report` using `src/gpu_simulate_test/analysis/report.py`
- [ ] T066 [US6] Update `run_state.json` with report outputs and print report path in `src/gpu_simulate_test/vidur_cli/stages.py`
- [ ] T067 [US6] Ensure `report/summary.md` includes arrival kind + CPU overhead status (warn if disabled) in `src/gpu_simulate_test/vidur_cli/stages.py`
- [ ] T068 [US6] Update report examples and expected artifacts in `specs/004-vidur-cli/quickstart.md`

**Checkpoint**: A full end-to-end run can produce a human-readable report.

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T069 [P] Improve CLI `--help` text and examples in `src/gpu_simulate_test/cli/vidur_cli.py`
- [ ] T070 Reconcile the implemented artifacts with the contract docs in `specs/004-vidur-cli/contracts/artifacts.md`
- [ ] T071 [P] Add a manual smoke checklist doc for the workflow in `tests/manual/vidur_cli_smoke.md`
- [ ] T072 Add a v1 smoke checklist (commands + expected files) in `specs/004-vidur-cli/checklists/smoke.md`
- [ ] T073 Re-run `pixi install` once after wiring the console script/task and ensure `pyproject.toml` and `pixi.lock` are consistent

---

## Dependencies & Execution Order

### User Story Dependency Graph

```text
Phase 1 (Setup)
  ↓
Phase 2 (Foundational)
  ↓
US1 (resources, preflight) ─┬─→ US2 (configs list)
                            └─→ US3 (init-run) → US4 (trace) → US5 (profile/sim/real) → US6 (report)
```

### Parallel Opportunities (by story)

- **US1**: Resource resolution tasks in `src/gpu_simulate_test/vidur_cli/resources.py` can be split across independent keys (see T025, T026).
- **US2**: Group scanning vs printing are separable (T031–T033 vs T034–T035).
- **US3**: Run tag generation vs stage runner wiring can proceed in parallel (T037 vs T040–T043).
- **US4**: Trace validation/meta vs compatibility artifacts can proceed in parallel (T045–T046 vs T050).
- **US5**: Token-length replay helper module can be implemented in parallel with profile/sim runners (T058 vs T056–T057).
- **US6**: Report generation + summary requirements can be implemented in parallel with run_state updates (T065 vs T066–T067).

---

## Parallel Examples (per User Story)

### US1

```text
In parallel:
- T025 (models/datasets root resolution in src/gpu_simulate_test/vidur_cli/resources.py)
- T026 (workspace_root resolution in src/gpu_simulate_test/vidur_cli/resources.py)
```

### US2

```text
In parallel:
- T031 (group discovery in src/gpu_simulate_test/vidur_cli/search_path.py)
- T032 (preset listing in src/gpu_simulate_test/vidur_cli/search_path.py)
```

### US3

```text
In parallel:
- T037 (run tag generation in src/gpu_simulate_test/vidur_cli/run_state.py)
- T041 (resources.json snapshot writing in src/gpu_simulate_test/vidur_cli/stages.py)
```

### US4

```text
In parallel:
- T045 (trace CSV validation in src/gpu_simulate_test/vidur_cli/trace.py)
- T050 (trace_lengths.csv / trace_intervals.csv compatibility outputs in src/gpu_simulate_test/vidur_cli/trace.py)
```

### US5

```text
In parallel:
- T056 (profile runner in src/gpu_simulate_test/vidur_cli/stages.py)
- T058 (token-length replay helpers in src/gpu_simulate_test/vidur_cli/real_runner.py)
```

### US6

```text
In parallel:
- T065 (report generation under src/gpu_simulate_test/analysis/report.py)
- T067 (summary.md additions in src/gpu_simulate_test/vidur_cli/stages.py)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Run the US1 independent test via `pixi run vidur-cli resources show`

### Incremental Delivery

1. Setup + Foundational → foundation ready
2. Add US1 → validate independently (MVP)
3. Add US2 → validate independently
4. Add US3 → validate independently
5. Add US4 → validate independently
6. Add US5 → validate independently
7. Add US6 → validate independently
