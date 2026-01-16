---
description: "Task list for implementing reliable Vidur MLP profiling for driver-launched kernels"
---

# Tasks: Reliable Vidur MLP profiling for driver-launched kernels

**Input**: Design documents from `specs/005-vidur-mlp-cuda-driver/`  
**Prerequisites**: `specs/005-vidur-mlp-cuda-driver/plan.md`, `specs/005-vidur-mlp-cuda-driver/spec.md`, `specs/005-vidur-mlp-cuda-driver/research.md`, `specs/005-vidur-mlp-cuda-driver/data-model.md`, `specs/005-vidur-mlp-cuda-driver/contracts/`, `specs/005-vidur-mlp-cuda-driver/quickstart.md`  
**Reference**: `context/issues/known/issue-vidur-mlp-profiling-misses-cuda-driver-kernels.md`

**Tests**: Automated regression tests are REQUIRED by `specs/005-vidur-mlp-cuda-driver/spec.md` (FR-008) and are included under User Story 3.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `- [ ] T### [P?] [US#?] Description with file path`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[US#]**: Which user story this task belongs to (e.g., `[US1]`)
- **File paths are required** in every task description

## Path Conventions

- Single project: `src/`, `tests/` at repository root
- Vidur profiling wrappers: `src/gpu_simulate_test/vidur_ext/`
- CLI entrypoints: `src/gpu_simulate_test/cli/`
- Feature docs live under `specs/005-vidur-mlp-cuda-driver/`

---

## Implementation Guides

- `context/tasks/working/005-vidur-mlp-cuda-driver/impl-phase-1-setup.md`
- `context/tasks/working/005-vidur-mlp-cuda-driver/impl-phase-2-foundational.md`
- `context/tasks/working/005-vidur-mlp-cuda-driver/impl-phase-3-us1-mlp-timings.md`
- `context/tasks/working/005-vidur-mlp-cuda-driver/impl-phase-4-us2-validation.md`
- `context/tasks/working/005-vidur-mlp-cuda-driver/impl-phase-5-us3-regression-tests.md`
- `context/tasks/working/005-vidur-mlp-cuda-driver/impl-phase-6-polish.md`
- `context/tasks/working/005-vidur-mlp-cuda-driver/impl-integrate-phases.md`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Ensure the repo environment and submodules match the feature’s assumptions.

- [ ] T001 Initialize required submodules via `git submodule update --init --recursive` (see `.gitmodules`)
- [ ] T002 Materialize the Pixi environment with `pixi install` (inputs: `pyproject.toml`, `pixi.lock`)
- [ ] T003 Refresh Codex agent context after finalizing docs via `.specify/scripts/bash/update-agent-context.sh` (script: `.specify/scripts/bash/update-agent-context.sh`)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Wire run configuration + plumbing needed by ALL user stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T004 Add required `profiling.mlp.profile_method` + default validation/fallback knobs to `configs/vidur_profiling/bundle.yaml`
- [ ] T005 [P] Add required `profiling.mlp.profile_method` + default validation/fallback knobs to `configs/paper_fidelity/profile.yaml`
- [ ] T006 [P] Add required `profiling.mlp.profile_method` + default validation/fallback knobs to `configs/compare_vidur_real/vidur_profile.yaml`
- [ ] T007 [P] Add consumer-side defaults for `vidur.validation.mlp.*` (mode + thresholds) to `configs/compare_vidur_real/vidur/default.yaml`
- [ ] T008 Update `VidurProfileInputs` to require explicit MLP method selection + carry validation/fallback settings in `src/gpu_simulate_test/vidur_ext/profile_runner.py`
- [ ] T009 Use `VidurProfileInputs.mlp_profile_method` to pass `--profile_method ...` explicitly into the MLP profiler subprocess in `src/gpu_simulate_test/vidur_ext/profile_runner.py`
- [ ] T010 Update `vidur-profiling-bundle` config parsing to read `profiling.mlp.*` and pass into `VidurProfileInputs` in `src/gpu_simulate_test/vidur_ext/profiling_bundle.py`
- [ ] T011 [P] Update `paper-fidelity profile` config parsing to read `profiling.mlp.*` and pass into `VidurProfileInputs` in `src/gpu_simulate_test/paper_fidelity/profiling.py`
- [ ] T012 [P] Update `vidur-cli svr profile` to read `profiling.mlp.*` and pass into `VidurProfileInputs` in `src/gpu_simulate_test/vidur_cli/stages.py`
- [ ] T013 [P] Update `vidur-profile` CLI wiring to pass `profiling.mlp.profile_method` into `VidurProfileInputs` in `src/gpu_simulate_test/cli/vidur_profile.py`

**Checkpoint**: Foundation ready — user story implementation can now begin.

---

## Phase 3: User Story 1 - Produce complete MLP timings (Priority: P1) 🎯 MVP

**Goal**: Ensure record-function-based attribution includes driver-launched kernels so staged `mlp.csv` has complete, non-missing core timing targets.

**Independent Test**: Run profiling + staging on a workload known to include driver-launched kernels and verify staged `mlp.csv` has 0 missing values for core `time_stats.*.{min,max,mean,median}` targets.

### Implementation for User Story 1

- [ ] T014 [P] [US1] Implement correlation-based `RecordFunctionTracerV2` supporting `cuda_runtime` + `cuda_driver` launches in `src/gpu_simulate_test/vidur_ext/record_function_tracer_v2.py`
- [ ] T015 [P] [US1] Monkey-patch `vidur.profiling.mlp.mlp_wrapper.RecordFunctionTracer` to `RecordFunctionTracerV2` when `--profile_method record_function` is selected in `src/gpu_simulate_test/vidur_ext/vidur_profiling_mlp_main.py`
- [ ] T016 [US1] Remove blanket NaN→0 staging for `time_stats.*` columns (stop `fillna(0.0)` masking) in `src/gpu_simulate_test/vidur_ext/profile_runner.py`
- [ ] T017 [US1] Manually profile a known driver-launch workload in strict mode and confirm staged `mlp.csv` has no missing core targets (commands: `specs/005-vidur-mlp-cuda-driver/quickstart.md`)

**Checkpoint**: User Story 1 is functional: record-function profiling produces complete timings for driver-launched kernels.

---

## Phase 4: User Story 2 - Fail fast on missing or suspicious data (Priority: P2)

**Goal**: Detect missing or suspiciously zero-heavy timing targets at both staging and consumption; fail fast by default with actionable remediation; allow opt-in fallback.

**Independent Test**: Provide an input profiling dataset with missing values and confirm staging/validation fails with an actionable error message.

### Implementation for User Story 2

- [ ] T018 [P] [US2] Implement `validate_mlp_csv` + `MlpValidationResult` (missing cells, missing columns, zero-heavy detection, strict/non-strict) in `src/gpu_simulate_test/vidur_ext/mlp_validation.py`
- [ ] T019 [US2] Enforce MLP validation during staging (strict by default; missing always fails; zero-heavy fails/warns per mode) with remediation messages in `src/gpu_simulate_test/vidur_ext/profile_runner.py`
- [ ] T020 [US2] Implement opt-in automatic fallback rerun on validation failure (`fallback.enabled`, `fallback.method`) and record the final method used in `src/gpu_simulate_test/vidur_ext/profile_runner.py`
- [ ] T021 [US2] Apply the same staging validation when compute profiling is skipped due to existing outputs (validate existing `mlp.csv` before returning) in `src/gpu_simulate_test/vidur_ext/profile_runner.py`
- [ ] T022 [P] [US2] Embed `mlp_validation` + method/fallback provenance into `profiling_meta.json` and reconcile with `specs/005-vidur-mlp-cuda-driver/contracts/*.schema.json` in `src/gpu_simulate_test/vidur_ext/profiling_bundle.py`
- [ ] T023 [P] [US2] Embed `mlp_validation` + method/fallback provenance into paper-fidelity `profiling_meta.json` in `src/gpu_simulate_test/paper_fidelity/profiling.py`
- [ ] T024 [US2] Enforce consumption-side MLP validation (strict fail vs non-strict warn) in `src/gpu_simulate_test/vidur_ext/profiling_root.py`
- [ ] T025 [US2] Extend `ProfilingRootLayout` with MLP validation settings (mode + thresholds) in `src/gpu_simulate_test/vidur_ext/profiling_root.py`
- [ ] T026 [US2] Add MLP validation settings to `VidurSimInputs` and pass through to `ProfilingRootLayout` in `src/gpu_simulate_test/vidur_ext/sim_runner.py`
- [ ] T027 [P] [US2] Thread consumer validation config (`vidur.validation.mlp.*`) into `VidurSimInputs` in `src/gpu_simulate_test/cli/vidur_sim.py`
- [ ] T028 [P] [US2] Thread consumer validation config (`vidur.validation.mlp.*`) into `VidurSimInputs` for `vidur-cli svr sim` in `src/gpu_simulate_test/vidur_cli/stages.py`

**Checkpoint**: Bad profiling roots fail fast (or warn in non-strict) at both staging and consumption, with clear remediation (including fallback guidance).

---

## Phase 5: User Story 3 - Prevent regressions (Priority: P3)

**Goal**: Add automated checks covering both runtime-launched and driver-launched kernel traces so attribution/validation behavior does not regress.

**Independent Test**: Run `pixi run pytest -q` and confirm the new unit tests pass on CPU-only hosts.

### Tests for User Story 3 ⚠️

- [ ] T029 [P] [US3] Add unit tests for `RecordFunctionTracerV2` covering runtime + driver launch paths with synthetic traces in `tests/unit/test_vidur_record_function_tracer_v2.py`
- [ ] T030 [P] [US3] Add unit tests for `validate_mlp_csv` (missing cells, missing columns, zero-heavy strict vs non-strict) in `tests/unit/test_mlp_validation.py`
- [ ] T031 [P] [US3] Add unit tests ensuring `validate_profiling_root` enforces MLP validation (strict fails; non-strict warns) in `tests/unit/test_profiling_root_mlp_validation.py`
- [ ] T032 [US3] Run the unit test suite in Pixi (`pixi run pytest -q`) and ensure new tests pass (tests live under `tests/unit/`)

**Checkpoint**: Regression coverage exists for both attribution and validation logic.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Update repo workflows and documentation impacted by explicit method selection and validation.

- [ ] T033 [P] Update manual smoke test to pass explicit `profiling.mlp.profile_method` overrides in `tests/manual/test_vidur_profiling_bundle_smoke.py`
- [ ] T034 [P] Update manual smoke test to pass explicit `profiling.mlp.profile_method` overrides in `tests/manual/test_vidur_profile_smoke.py`
- [ ] T035 [P] Update manual smoke test to pass explicit `profiling.mlp.profile_method` overrides in `tests/manual/test_paper_fidelity_profile_smoke.py`
- [ ] T036 [P] Update the profiling bundle helper script to pass explicit `profiling.mlp.profile_method` (and optional fallback knobs) in `scripts/run_vidur_profiling_llama2_7b.sh`
- [ ] T037 Update the known-issue writeup with fix status + new remediation guidance in `context/issues/known/issue-vidur-mlp-profiling-misses-cuda-driver-kernels.md`
- [ ] T038 Re-run the workflow examples and refresh any gotchas in `specs/005-vidur-mlp-cuda-driver/quickstart.md`

---

## Dependencies & Execution Order

### User Story Dependency Graph

```text
Phase 1 (Setup)
  ↓
Phase 2 (Foundational)
  ↓
US1 (P1: attribution works) → US2 (P2: validation + fallback + provenance) → US3 (P3: regression tests)
```

### Parallel Opportunities (by story)

- **Foundational**: Config changes across `configs/` files (T004–T007) and call-site plumbing (T010–T013) can be split across owners.
- **US1**: Tracer implementation vs wrapper monkey-patch can proceed in parallel (T014 vs T015).
- **US2**: Meta/provenance updates can proceed in parallel once validation result shape is finalized (T022 vs T023); sim vs vidur-cli consumer plumbing can proceed in parallel (T027 vs T028).
- **US3**: Tracer tests vs validation tests vs profiling-root tests can proceed in parallel (T029–T031).
- **Polish**: Manual smoke and script updates are independent (T033–T036).

---

## Parallel Examples (per story)

### User Story 1

```bash
Task: "Implement RecordFunctionTracerV2 in src/gpu_simulate_test/vidur_ext/record_function_tracer_v2.py"
Task: "Patch vidur_profiling_mlp_main to use the tracer in src/gpu_simulate_test/vidur_ext/vidur_profiling_mlp_main.py"
```

### User Story 2

```bash
Task: "Implement validate_mlp_csv in src/gpu_simulate_test/vidur_ext/mlp_validation.py"
Task: "Embed mlp_validation into profiling_meta.json in src/gpu_simulate_test/vidur_ext/profiling_bundle.py"
Task: "Embed mlp_validation into profiling_meta.json in src/gpu_simulate_test/paper_fidelity/profiling.py"
```

### User Story 3

```bash
Task: "Add tracer tests in tests/unit/test_vidur_record_function_tracer_v2.py"
Task: "Add MLP validation tests in tests/unit/test_mlp_validation.py"
Task: "Add profiling-root validation tests in tests/unit/test_profiling_root_mlp_validation.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL — blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Run the independent test for US1 (driver-launch workload; 0 missing core targets)

### Incremental Delivery

1. Setup + Foundational → baseline wiring
2. US1 attribution fix → verify on known driver-launch workload
3. US2 validation + fallback + provenance → verify fail-fast + fallback behavior
4. US3 regression tests → keep behavior stable across refactors
