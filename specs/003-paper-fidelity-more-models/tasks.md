---

description: "Task list for implementing Paper-fidelity more models"
---

# Tasks: Paper-fidelity more models

**Input**: Design documents from `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/`  
**Prerequisites**: `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/plan.md` (required), `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/spec.md` (required), `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/research.md`, `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/data-model.md`, `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/contracts/openapi.yaml`, `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/quickstart.md`

**Tests**: Automated tests were not explicitly requested in the spec; tasks below focus on implementation + manual/CLI validation steps.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- Single project: `src/`, `tests/` at repository root
- All file paths in tasks are absolute for copy/paste execution.

## Implementation Guides

- `context/tasks/working/003-paper-fidelity-more-models/impl-phase-1-setup.md`
- `context/tasks/working/003-paper-fidelity-more-models/impl-phase-2-foundational.md`
- `context/tasks/working/003-paper-fidelity-more-models/impl-phase-3-us1-scenarios.md`
- `context/tasks/working/003-paper-fidelity-more-models/impl-phase-4-us2-profiling.md`
- `context/tasks/working/003-paper-fidelity-more-models/impl-phase-5-us3-static-report.md`
- `context/tasks/working/003-paper-fidelity-more-models/impl-phase-6-us4-dynamic-report.md`
- `context/tasks/working/003-paper-fidelity-more-models/impl-phase-7-us5-matrix-runner.md`
- `context/tasks/working/003-paper-fidelity-more-models/impl-phase-8-us6-failure-records.md`
- `context/tasks/working/003-paper-fidelity-more-models/impl-phase-9-polish.md`
- `context/tasks/working/003-paper-fidelity-more-models/impl-integrate-phases.md`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add shared schema/paths helpers used across multiple user stories

- [ ] T001 Create failure record schema + JSON helpers in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/failure_record.py`
- [ ] T002 [P] Create matrix manifest schema + JSON writer in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/matrix_manifest.py`
- [ ] T003 [P] Add matrix output path helpers (manifest dir + failures dir) to `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/paths.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core utilities required before implementing story work (GPU gating, scenario validation, blocker categorization)

**⚠️ CRITICAL**: No user story work should begin until this phase is complete

- [ ] T004 Add `count_visible_gpus()` helper to `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/env_guard.py` (parse `CUDA_VISIBLE_DEVICES` after `apply_cuda_visible_devices_from_gsim`)
- [ ] T005 Add scenario preflight validation + custom exceptions in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/validation.py` (validate `scenario.model.model_ref`, `scenario.trace_source.path`, and required GPUs from TP/PP)
- [ ] T006 Add blocker categorization helper in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/failure_record.py` (map exception/stderr → `insufficient GPUs|OOM|missing model files|unsupported model|unknown`)

**Checkpoint**: Foundation ready — user story implementation can now begin

---

## Phase 3: User Story 1 - Add paper-fidelity scenarios for paper models (Priority: P1) 🎯 MVP

**Goal**: Add first-class scenarios for InternLM-20B, LLaMA2-70B, and Qwen-72B (exclude Qwen3-0.6B) so `paper-fidelity trace` works for each.

**Independent Test**: A developer can select each new scenario and generate a valid canonical trace for both static and dynamic workloads.

- [ ] T007 [P] [US1] Add InternLM-20B scenario config in `/data1/huangzhe/code/gpu-simulate-test/configs/paper_fidelity/scenario/internlm_20b_arxiv.yaml`
- [ ] T008 [P] [US1] Add LLaMA2-70B scenario config in `/data1/huangzhe/code/gpu-simulate-test/configs/paper_fidelity/scenario/llama2_70b_arxiv.yaml`
- [ ] T009 [P] [US1] Add Qwen-72B scenario config in `/data1/huangzhe/code/gpu-simulate-test/configs/paper_fidelity/scenario/qwen_72b_arxiv.yaml`
- [ ] T010 [US1] Add trace preflight validation (trace source exists, friendly error) in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/cli/paper_fidelity.py`

**Manual validation (commands live in docs)**: `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/quickstart.md`

---

## Phase 4: User Story 2 - Generate host-matched profiling roots per model (Priority: P1)

**Goal**: Support host profiling per new scenario, always including CPU overhead microbenchmarks for the acceptance matrix, and record failures when profiling cannot complete.

**Independent Test**: For each model scenario, host profiling completes and produces a Vidur-compatible profiling root with the expected structure.

- [ ] T011 [US2] Add profiling preflight validation (model assets exist, GPU sufficiency) in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/profiling.py`
- [ ] T012 [US2] Write `failure_record.json` on profiling failures (include attempted command/hydra overrides + blocker category) in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/profiling.py`
- [ ] T013 [US2] Print failure record path on `paper-fidelity profile` failure in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/cli/paper_fidelity.py`

**Manual validation (commands live in docs)**: `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/quickstart.md`

---

## Phase 5: User Story 3 - Produce a static paper-fidelity report for each model (Priority: P2)

**Goal**: Ensure a static repro at `--scale small` produces a scored, self-contained report bundle per model.

**Independent Test**: A static run per model produces a report directory containing a human-readable summary and machine-readable scoring outputs.

- [ ] T014 [US3] Snapshot profiling metadata into report inputs (copy `profiling_meta.json` when present) in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/cli/paper_fidelity.py`
- [ ] T015 [US3] Update static repro documentation/examples for all paper models in `/data1/huangzhe/code/gpu-simulate-test/docs/tutorial/howto/tut-paper-fidelity-static-and-dynamic.md`

---

## Phase 6: User Story 4 - Produce a dynamic paper-fidelity report for each model (Priority: P2)

**Goal**: Ensure a dynamic repro at `--scale small` produces a scored report bundle per model including capacity discovery and a timed trace snapshot.

**Independent Test**: A dynamic run per model produces a report directory and includes trace and capacity artifacts needed for debugging.

- [ ] T016 [US4] Normalize dynamic `trace_meta.json` schema (include `trace_source` + `artifacts.trace_csv` like `paper-fidelity trace`) in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/cli/paper_fidelity.py`
- [ ] T017 [US4] Update dynamic repro documentation/examples for all paper models in `/data1/huangzhe/code/gpu-simulate-test/docs/tutorial/howto/tut-paper-fidelity-static-and-dynamic.md`

---

## Phase 7: User Story 5 - Run the full model/workload matrix with one repeatable procedure (Priority: P3)

**Goal**: Provide a single `paper-fidelity matrix` procedure that runs (profile + static repro + dynamic repro) for the required paper models at `--scale small`, writes a per-matrix manifest, and points to all outputs.

**Independent Test**: A single documented procedure produces per-model/per-workload outputs and a clear manifest summarizing successes and where reports are located.

- [ ] T018 [US5] Add paper-model scenario constants and explicit Qwen3-0.6B exclusion in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/paper_models.py`
- [ ] T019 [US5] Implement matrix runner orchestration (profile → repro static/dynamic) in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/matrix.py`
- [ ] T020 [US5] Add `matrix` subcommand + flags (`--scale`, `--scenarios`, `--workloads`, `--include-cpu-overhead`, `--run-id`, `--stop-on-failure`) in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/cli/paper_fidelity.py`
- [ ] T021 [US5] Write per-matrix outputs (`manifest.json`, `failures/*.json`) under `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/paper_models_matrix_<run_id>/` via `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/matrix_manifest.py`
- [ ] T022 [US5] Update matrix quickstart to match implemented CLI flags and output paths in `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/quickstart.md`

---

## Phase 8: User Story 6 - Diagnose and record failures consistently (Priority: P3)

**Goal**: When runs fail (missing assets, insufficient GPUs, unsupported model, OOM), write structured failure records with blocker categorization so failures are debuggable without reruns.

**Independent Test**: Intentionally triggering a failure produces a failure record containing the attempted action, the error message, and the blocker category.

- [ ] T023 [US6] Add repro failure record writing (create report dir + write `failure_record.json` on exception) in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/cli/paper_fidelity.py`
- [ ] T024 [US6] Ensure matrix runner writes failure records per action (profile/static/dynamic) and continues unless `--stop-on-failure` in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/matrix.py`
- [ ] T025 [US6] Document failure record schema + blocker categories in `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/quickstart.md`

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, UX, and consistency improvements across stories

- [ ] T026 [P] Update scenario tutorial to reference paper-model scenarios + matrix runner in `/data1/huangzhe/code/gpu-simulate-test/docs/tutorial/in-depth/adv-tut-add-paper-fidelity-scenario.md`
- [ ] T027 [P] Update top-level docs entrypoints for the matrix workflow in `/data1/huangzhe/code/gpu-simulate-test/README.md`
- [ ] T028 [P] Add a short runbook for reading `manifest.json` + `failure_record.json` in `/data1/huangzhe/code/gpu-simulate-test/docs/runbooks/paper_fidelity_matrix.md`
- [ ] T029 Run the quickstart end-to-end and refresh expected outputs/examples in `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/quickstart.md`

---

## Dependencies & Execution Order

### User Story Dependency Graph

```text
US1 (scenarios + trace)
  └─> US2 (host profiling)
        ├─> US3 (static reports)
        ├─> US4 (dynamic reports)
        └─> US5 (matrix runner) ──> US6 (failure records hardening)
```

Notes:
- US2 depends on US1 (new scenario configs must exist).
- US3/US4 depend on US2 (host profiling root required for meaningful sim-vs-real runs).
- US5 depends on US1–US4 (it orchestrates profile + repro).
- US6 can be developed alongside US5, but must be complete before final acceptance (failure transparency requirements).

### Parallel Opportunities (Examples)

- Scenario YAMLs for different models can be implemented in parallel: T007–T009.
- Matrix infrastructure can be implemented in parallel across files: T002, T003, T018–T021.
- Docs updates can be parallelized: T015, T017, T022, T025–T028.

---

## Parallel Execution Examples (Per User Story)

### US1

```text
T007 (internlm_20b_arxiv.yaml) + T008 (llama2_70b_arxiv.yaml) + T009 (qwen_72b_arxiv.yaml) can run in parallel.
```

### US2

```text
T011 (profiling preflight) and T012 (profiling failure record writing) can be developed in parallel if they touch different functions in profiling.py.
```

### US3

```text
T014 (code change) and T015 (docs update) can run in parallel.
```

### US4

```text
T016 (code change) and T017 (docs update) can run in parallel.
```

### US5

```text
T018 (paper_models.py) + T019 (matrix.py) + T020 (CLI wiring) + T021 (manifest writer) can be split across different files and run in parallel.
```

### US6

```text
T023 (repro failure records) and T024 (matrix failure records) can run in parallel; both depend on the shared failure_record.py schema (T001/T006).
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 + Phase 2 (shared scaffolding)
2. Complete Phase 3 (US1: scenarios + trace validation)
3. Validate traces for all three scenarios per `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/quickstart.md`

### Incremental Delivery

1. US1 → validate trace generation
2. US2 → validate profiling roots (with CPU overhead)
3. US3 → validate static report bundles
4. US4 → validate dynamic report bundles
5. US5 + US6 → validate matrix runner outputs + failure records
