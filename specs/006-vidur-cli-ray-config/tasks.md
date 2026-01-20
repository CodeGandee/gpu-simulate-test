---

description: "Task list for implementing Ray runtime configuration in vidur-cli"
---

# Tasks: Vidur CLI Ray runtime configuration (env > config > defaults)

**Input**: Design documents from `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/`  
**Prerequisites**: `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/plan.md`, `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/spec.md`  

**Tests**: INCLUDED (required by SC-005 in `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/spec.md`).  

## Format: `- [ ] T### [P?] [US?] Description with absolute file path`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[US#]**: User story label (required only inside user story phases)
- Every task must include at least one absolute path under `/data1/huangzhe/code/gpu-simulate-test/`

## Implementation Guides

- `context/tasks/working/006-vidur-cli-ray-config/impl-phase-1-setup.md`
- `context/tasks/working/006-vidur-cli-ray-config/impl-phase-2-foundational.md`
- `context/tasks/working/006-vidur-cli-ray-config/impl-phase-3-us1-ray-settings.md`
- `context/tasks/working/006-vidur-cli-ray-config/impl-phase-4-us2-env-precedence.md`
- `context/tasks/working/006-vidur-cli-ray-config/impl-phase-5-us3-no-ray-profiling.md`
- `context/tasks/working/006-vidur-cli-ray-config/impl-phase-6-us4-validation.md`
- `context/tasks/working/006-vidur-cli-ray-config/impl-phase-7-us5-docs.md`
- `context/tasks/working/006-vidur-cli-ray-config/impl-phase-8-polish.md`
- `context/tasks/working/006-vidur-cli-ray-config/impl-integrate-phases.md`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Ensure the dev environment + submodules needed for Ray/Vidur/Sarathi work.

- [X] T001 Ensure Pixi environment is installed and usable (run `pixi install`; verify `/data1/huangzhe/code/gpu-simulate-test/.pixi/envs/default/` exists and Ray imports inside Pixi)
- [X] T002 Ensure git submodules are initialized for Vidur + Sarathi (run `git submodule update --init --recursive`; verify `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur/` and `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/sarathi-serve/` exist)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared plumbing needed by all stories (configs + helper module).

- [X] T003 Create Ray config group directory `/data1/huangzhe/code/gpu-simulate-test/configs/compare_vidur_real/ray/` and file `/data1/huangzhe/code/gpu-simulate-test/configs/compare_vidur_real/ray/default.yaml`
- [X] T004 Populate `/data1/huangzhe/code/gpu-simulate-test/configs/compare_vidur_real/ray/default.yaml` with `ray.env` keys (all default to `null`) and inline comments for supported settings
- [X] T005 Update Hydra defaults to include `ray: default` and add `profiling.compute.use_ray: true` in `/data1/huangzhe/code/gpu-simulate-test/configs/compare_vidur_real/vidur_profile.yaml`
- [X] T006 [P] Update Hydra defaults to include `ray: default` in `/data1/huangzhe/code/gpu-simulate-test/configs/compare_vidur_real/real_bench.yaml`
- [X] T007 [P] Update Hydra defaults to include `ray: default` in `/data1/huangzhe/code/gpu-simulate-test/configs/compare_vidur_real/vidur_sim.yaml`
- [X] T008 Create helper module skeleton in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/ray_runtime.py` (no `ray` import) with a public API placeholder: `SUPPORTED_RAY_ENV_KEYS`, `apply_ray_env_defaults(...)`, `write_ray_settings_json(...)`

**Checkpoint**: Foundation ready — user story work can begin.

---

## Phase 3: User Story 1 - Configure Ray runtime settings from `vidur-cli` config (Priority: P1) 🎯 MVP

**Goal**: Users can set a small supported set of Ray runtime settings via Hydra config for `vidur-cli` stages (profiling + real replay) without manual `export RAY_*`.

**Independent Test** (from spec): Run a Ray-using `vidur-cli` stage with no relevant `RAY_*` env vars set and with Ray settings provided via config; verify the stage reports the effective settings and completes successfully.

### Tests for User Story 1 (write first; fail before implementation)

- [X] T009 [P] [US1] Add unit tests for config-sourced env injection in `/data1/huangzhe/code/gpu-simulate-test/tests/unit/test_ray_runtime_config.py` (env unset + config provides all 3 supported keys ⇒ `os.environ` set + report sources=`configuration`)
- [X] T010 [P] [US1] Add unit test for config omissions/no-op in `/data1/huangzhe/code/gpu-simulate-test/tests/unit/test_ray_runtime_config.py` (config values `None` ⇒ no env injection + report sources=`default` with `effective_value=null`)

### Implementation for User Story 1

- [X] T011 [US1] Implement `apply_ray_env_defaults()` in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/ray_runtime.py` to apply **config > default** when env is unset and return a stable per-key report (`effective_value` stays `null` when not set by env/config)
- [X] T012 [P] [US1] Integrate Ray env application + report emission into profiling stage in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_cli/stages.py` (`run_profile()`), writing `ray_settings.json` under `Path(run_dir)/profile/` and printing a human-readable summary to stderr
- [X] T013 [P] [US1] Integrate Ray env application + report emission into real replay stage in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_cli/stages.py` (`run_real()`), writing `ray_settings.json` under `Path(run_dir)/real/` when the backend uses Ray (e.g., Sarathi)
- [X] T014 [US1] Record the `ray_settings.json` absolute path in the stage artifact section of `run_state.json` within `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_cli/stages.py` (add non-breaking fields under `artifacts.profile` / `artifacts.real`)
- [X] T015 [P] [US1] Add a manual smoke procedure doc at `/data1/huangzhe/code/gpu-simulate-test/tests/manual/vidur_cli_ray_settings_smoke.md` (commands + expected “effective settings report” output for `svr profile` and `svr real`)

**Checkpoint**: US1 complete when users can run `svr profile`/`svr real` with config-only Ray settings and see the effective settings report.

---

## Phase 4: User Story 2 - Respect user-provided `RAY_*` environment variables (Priority: P1)

**Goal**: If a supported `RAY_*` env var is already set, `vidur-cli` never overrides it; precedence is env > config per setting.

**Independent Test** (from spec): Set a supported `RAY_*` env var to a non-default value, run a Ray-using stage with a conflicting config value, and verify the effective settings report shows the environment value is used.

### Tests for User Story 2 (write first; fail before implementation)

- [X] T016 [P] [US2] Add unit test in `/data1/huangzhe/code/gpu-simulate-test/tests/unit/test_ray_runtime_config.py` for env > config precedence (env set + conflicting config ⇒ env preserved + report source=`environment`)
- [X] T017 [P] [US2] Add unit test in `/data1/huangzhe/code/gpu-simulate-test/tests/unit/test_ray_runtime_config.py` for mixed sourcing (some keys from env, some from config ⇒ per-key sources reflected correctly)

### Implementation for User Story 2

- [X] T018 [US2] Update `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/ray_runtime.py` so each supported key applies precedence **environment > configuration > default** (never override an existing env var)
- [X] T019 [US2] Ensure the stderr report emitted from `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_cli/stages.py` clearly shows per-key sources (`environment` vs `configuration` vs `default`)

**Checkpoint**: US2 complete when conflicting env/config always resolves to env and is visibly reported.

---

## Phase 5: User Story 3 - Optionally avoid Ray for compute profiling (Priority: P2)

**Goal**: Users can disable Ray usage for Vidur compute profiling (MLP/attention) in a supported single-GPU configuration; if work can’t run without Ray, generate downstream-compatible fallback outputs and clearly indicate it.

**Independent Test** (from spec): Run compute profiling with “disable Ray for compute profiling” enabled and verify it completes (for the supported scope) and produces outputs that downstream steps can consume.

### Tests for User Story 3 (write first; fail before implementation)

- [X] T020 [P] [US3] Add unit tests for no-Ray configuration gating in `/data1/huangzhe/code/gpu-simulate-test/tests/unit/test_vidur_profile_no_ray.py` (unsupported cases like `num_gpus>1` or `tensor_parallel_size>1` raise `UserFacingError` with an actionable hint)
- [X] T021 [P] [US3] Add unit test in `/data1/huangzhe/code/gpu-simulate-test/tests/unit/test_vidur_profile_no_ray.py` that `profiling.compute.use_ray=false` with CPU overhead enabled is rejected (or explicitly skipped) so the stage does not start Ray (per `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/spec.md` FR-014)

### Implementation for User Story 3

- [X] T022 [US3] Read `profiling.compute.use_ray` from the composed config and enforce no-Ray preconditions in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_cli/stages.py` (`run_profile()`), before invoking profiling runners
- [X] T023 [US3] Extend `VidurProfileInputs` with `compute_use_ray: bool` (and plumb it through) in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_ext/profile_runner.py`
- [X] T024 [US3] Implement the no-Ray compute profiling path in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_ext/profile_runner.py`: run MLP profiling sequentially via `vidur.profiling.mlp.mlp_wrapper.MlpWrapper` (single GPU), skip attention execution and always write attention fallback CSV via `_write_attention_fallback`, and clearly record fallback usage in `VidurProfileResult.extra`
- [X] T025 [P] [US3] Add manual smoke procedure doc at `/data1/huangzhe/code/gpu-simulate-test/tests/manual/vidur_cli_no_ray_compute_profiling_smoke.md` (commands + how to confirm Ray was not started + which outputs should exist)

**Checkpoint**: US3 complete when compute profiling can run with Ray disabled (supported scope) and downstream sim-vs-real steps can still consume the profiling root.

---

## Phase 6: User Story 4 - Fail fast on misconfiguration with actionable errors (Priority: P2)

**Goal**: Misconfigured Ray settings (env or config) fail fast before starting Ray, with clear, actionable errors.

**Independent Test** (from spec): Introduce a deliberate config/env mistake for a supported Ray setting and verify the stage exits quickly, without starting Ray, and prints an actionable error.

### Tests for User Story 4 (write first; fail before implementation)

- [X] T026 [P] [US4] Add unit tests for invalid config values in `/data1/huangzhe/code/gpu-simulate-test/tests/unit/test_ray_runtime_config.py` (negative bytes, proportion outside `(0,1]`, non-bool slow-storage) ⇒ `UserFacingError` that names the key + expected format
- [X] T027 [P] [US4] Add unit tests for invalid env values in `/data1/huangzhe/code/gpu-simulate-test/tests/unit/test_ray_runtime_config.py` (empty string, non-parseable values) ⇒ `UserFacingError` that names the env var + how to fix it

### Implementation for User Story 4

- [X] T028 [US4] Implement strict validation + unsupported-key rejection in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/ray_runtime.py` (reject unknown keys under `cfg.ray.env` and list the supported keys)
- [X] T029 [US4] Ensure `vidur-cli` applies validation before any Ray-starting imports in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_cli/stages.py` and only runs the Ray settings logic for Ray-using stages/backends

**Checkpoint**: US4 complete when invalid inputs fail fast reliably and never silently fall back to defaults.

---

## Phase 7: User Story 5 - Configure safely using documentation (Priority: P3)

**Goal**: Documentation explains supported settings, precedence rules, opt-in defaults, and provides a Docker-friendly example to avoid known memory spikes.

**Independent Test** (from spec): Locate docs for this feature and verify supported settings list + precedence + opt-in behavior + Docker-friendly example exist.

### Implementation for User Story 5

- [X] T030 [P] [US5] Update `/data1/huangzhe/code/gpu-simulate-test/context/issues/known/issue-vidur-ray-object-store-memory-spike-in-docker.md` to document the new `ray.env.*` config knobs, precedence rules, and at least one Docker-friendly example (no manual `export RAY_*`)
- [X] T031 [P] [US5] Update `/data1/huangzhe/code/gpu-simulate-test/configs/README.md` to describe the `ray` config group and list supported keys + opt-in default behavior
- [X] T032 [US5] Align `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/quickstart.md` with the final implemented behavior (especially no-Ray limitations + required flags)

**Checkpoint**: US5 complete when docs are sufficient for a new user to configure Ray safely.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Tighten quality across all stories (tests, docs consistency, cleanup).

- [X] T033 Run full unit test suite and fix regressions (`/data1/huangzhe/code/gpu-simulate-test/tests/`, command: `pixi run pytest`)
- [X] T034 [P] Ensure the Ray settings artifact schema matches implementation (`/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/contracts/ray_settings.schema.json`)
- [X] T035 [P] Add a brief changelog note in `/data1/huangzhe/code/gpu-simulate-test/context/plans/plan-vidur-cli-ray-runtime-config.md` linking to the implemented feature and describing user-visible changes

---

## Dependencies & Execution Order

### Phase Dependencies

- Setup (Phase 1) → Foundational (Phase 2) → US1 (Phase 3) → US2 (Phase 4) → US3 (Phase 5) → US4 (Phase 6) → US5 (Phase 7) → Polish (Phase 8)

### User Story Dependency Graph

```text
US1 (config-defined Ray settings + report) ─┬─> US2 (env precedence)
                                           ├─> US3 (no-Ray compute profiling)
                                           └─> US4 (fail-fast validation)
US5 (docs) depends on US1–US4 behavior being final
```

### Parallel Opportunities

- Phase 2: T006 and T007 can run in parallel (different config files).
- US1 tests (T009, T010) can run in parallel (same file, but separate cases; coordinate on merges).
- US2 tests (T016, T017) can run in parallel (same caveat).
- US3 and US4 can be developed in parallel after US1 if staffed, but both touch `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/ray_runtime.py` and `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_cli/stages.py`, so split carefully.

---

## Parallel Execution Examples

### User Story 1

```text
Task: "T009 [US1] Unit tests for config injection in /data1/huangzhe/code/gpu-simulate-test/tests/unit/test_ray_runtime_config.py"
Task: "T010 [US1] Unit tests for no-op behavior in /data1/huangzhe/code/gpu-simulate-test/tests/unit/test_ray_runtime_config.py"
Task: "T012 [US1] Integrate profile stage in /data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_cli/stages.py"
Task: "T013 [US1] Integrate real stage in /data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_cli/stages.py"
```

### User Story 2

```text
Task: "T016 [US2] Env-precedence unit test in /data1/huangzhe/code/gpu-simulate-test/tests/unit/test_ray_runtime_config.py"
Task: "T017 [US2] Mixed-sourcing unit test in /data1/huangzhe/code/gpu-simulate-test/tests/unit/test_ray_runtime_config.py"
```

### User Story 3

```text
Task: "T020 [US3] No-Ray gating tests in /data1/huangzhe/code/gpu-simulate-test/tests/unit/test_vidur_profile_no_ray.py"
Task: "T025 [US3] Manual smoke doc in /data1/huangzhe/code/gpu-simulate-test/tests/manual/vidur_cli_no_ray_compute_profiling_smoke.md"
```

### User Story 4

```text
Task: "T026 [US4] Invalid-config unit tests in /data1/huangzhe/code/gpu-simulate-test/tests/unit/test_ray_runtime_config.py"
Task: "T027 [US4] Invalid-env unit tests in /data1/huangzhe/code/gpu-simulate-test/tests/unit/test_ray_runtime_config.py"
```

### User Story 5

```text
Task: "T030 [US5] Update Docker issue doc in /data1/huangzhe/code/gpu-simulate-test/context/issues/known/issue-vidur-ray-object-store-memory-spike-in-docker.md"
Task: "T031 [US5] Update configs README in /data1/huangzhe/code/gpu-simulate-test/configs/README.md"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1–2 (environment + config group + helper skeleton)
2. Complete Phase 3 (US1) end-to-end
3. Validate US1 independently using `/data1/huangzhe/code/gpu-simulate-test/tests/manual/vidur_cli_ray_settings_smoke.md`

### Incremental Delivery

1. US1 → US2 (safe precedence) → US3 (no-Ray compute profiling) → US4 (strict validation) → US5 (docs)
2. Run `pixi run pytest` after each story phase to keep regressions small (`/data1/huangzhe/code/gpu-simulate-test/tests/`)
