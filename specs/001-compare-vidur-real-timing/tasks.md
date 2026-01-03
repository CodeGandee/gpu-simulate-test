# Tasks: Compare Vidur vs real Qwen3 A100 timing

**Input**: Design docs under `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/`  
**Prerequisites**: `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/plan.md` and `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/spec.md`

**Validation**: Every change MUST include at least one of manual/unit/integration validation under `/data1/huangzhe/code/gpu-simulate-test/tests/` (per the constitution). Prefer manual scripts for GPU/experiment-heavy flows.

**Organization**: Tasks are grouped by user story so each story is independently implementable and testable.

## Phase 1: Shared foundation (Hydra + artifacts)

- [X] T001 Create code skeleton per `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/plan.md` under `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/` (`cli/`, `config/`, `workloads/`, `real_bench/`, `vidur_ext/`, `analysis/`)
- [X] T002 [P] Create Hydra config tree under `/data1/huangzhe/code/gpu-simulate-test/configs/compare_vidur_real/` (root stage configs + groups: `model/`, `hardware/`, `workload/`, `backend/`, `vidur/`)
- [X] T003 [P] Add Pixi tasks in `/data1/huangzhe/code/gpu-simulate-test/pyproject.toml` for `workload-spec`, `real-bench`, `vidur-profile`, `vidur-sim`, `compare-runs` (each invokes a Hydra app module)
- [X] T004 Implement shared structured config dataclasses in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/config/` (paths, model, hardware, workload, profiling, reporting)
- [X] T005 Implement shared artifact helpers in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/io/` (write/read CSV + JSON, schema checks, stable run/workload ids)
- [X] T006 [P] Unit tests for artifact schema helpers in `/data1/huangzhe/code/gpu-simulate-test/tests/unit/test_artifact_schemas.py`

## User Story 1 (P1): Generate deterministic workload spec

**Goal**: Deterministically generate `prompts.jsonl`, `trace_lengths.csv`, `trace_intervals.csv`, and `workload_meta.json`.

**Independent Test**: Generate workload spec twice with same seed and verify identical traces and required columns.

### Validation (US1)

- [X] T101 [P] Unit test determinism for `trace_intervals.csv` and required columns in `/data1/huangzhe/code/gpu-simulate-test/tests/unit/test_workload_spec_determinism.py`
- [X] T102 [P] Manual smoke run script in `/data1/huangzhe/code/gpu-simulate-test/tests/manual/test_workload_spec_smoke.py` (writes to `tmp/`)

### Implementation (US1)

- [X] T103 [P] Implement prompts loader/writer in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/workloads/prompts.py`
- [X] T104 [P] Implement token length extraction (tokenizer selection + counting) in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/workloads/token_lengths.py`
- [X] T105 Implement deterministic arrival schedule generation in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/workloads/arrival_schedule.py` (nanosecond integer outputs)
- [X] T106 Implement Hydra entrypoint in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/cli/workload_spec.py` (uses `/data1/huangzhe/code/gpu-simulate-test/configs/compare_vidur_real/workload_spec.yaml`)
- [X] T107 Ensure `workload_meta.json` is written and includes config + provenance per `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/contracts/workload_spec.md`

## User Story 2 (P1): Collect real A100 timing metrics

**Goal**: Run a real benchmark driven by workload spec and emit `request_metrics.csv` + `token_metrics.csv` with consistent schema for both backends.

**Independent Test**: Run a 2–5 request workload and validate output files and monotonic timestamp ordering.

### Validation (US2)

- [X] T201 [P] Manual smoke run script in `/data1/huangzhe/code/gpu-simulate-test/tests/manual/test_real_bench_smoke.py` (A100-required; writes to `tmp/real_runs/`)
- [X] T202 [P] Unit test metrics schema validation in `/data1/huangzhe/code/gpu-simulate-test/tests/unit/test_real_metrics_schema.py`

### Implementation (US2)

- [X] T203 Implement trace replay/scheduler helper in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/real_bench/replay.py` (uses `trace_intervals.csv`)
- [X] T204 [P] Implement backend adapter for `transformers` in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/real_bench/backends/transformers_backend.py`
- [X] T205 [P] Implement backend adapter for `sarathi` in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/real_bench/backends/sarathi_backend.py`
- [X] T206 Implement metrics recorder in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/real_bench/metrics.py` (writes per `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/contracts/request_metrics.md` and `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/contracts/token_metrics.md`)
- [X] T207 Implement Hydra entrypoint in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/cli/real_bench.py` (supports backend selection; writes `run_meta.json`)

## User Story 3 (P1): Run Vidur simulation from repo root

**Goal**: Run Vidur from repo root, using explicit profiling root and workload spec, producing standardized metrics artifacts.

**Independent Test**: Run simulator on tiny workload with a valid profiling root; verify outputs and fail-fast behavior for missing inputs.

### Validation (US3)

- [X] T301 [P] Manual smoke run script in `/data1/huangzhe/code/gpu-simulate-test/tests/manual/test_vidur_sim_smoke.py` (requires profiling data; writes to `tmp/vidur_runs/`)
- [X] T302 [P] Unit test fail-fast errors for missing profiling root/files in `/data1/huangzhe/code/gpu-simulate-test/tests/unit/test_vidur_sim_prereqs.py`

### Implementation (US3)

- [X] T303 Implement profiling-root validation helpers in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_ext/profiling_root.py`
- [X] T304 Implement Vidur simulation runner wrapper in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_ext/sim_runner.py` (sets required Vidur config paths from Hydra config)
- [X] T305 Implement Hydra entrypoint in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/cli/vidur_sim.py` (writes standardized `request_metrics.csv`, `token_metrics.csv`, `run_meta.json`)

## User Story 4 (P1): Produce a comparison report

**Goal**: Generate `summary.md` + percentile tables + plots comparing one real run vs one sim run, with early-stop alignment.

**Independent Test**: Compare two existing run dirs (CPU-only) and verify percentiles + at least one plot per metric family.

### Validation (US4)

- [X] T401 [P] Unit test early-stop alignment logic in `/data1/huangzhe/code/gpu-simulate-test/tests/unit/test_compare_alignment.py`
- [X] T402 [P] Manual smoke run script in `/data1/huangzhe/code/gpu-simulate-test/tests/manual/test_compare_runs_smoke.py` (CPU-only; writes to `tmp/comparisons/`)

### Implementation (US4)

- [X] T403 Implement metrics loaders (real+sim) in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/analysis/load_metrics.py`
- [X] T404 Implement alignment + percentile computation in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/analysis/compare.py` (truncate sim tokens using `num_decode_tokens_actual`)
- [X] T405 [P] Implement plotting utilities in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/analysis/plots.py`
- [X] T406 Implement report writer in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/analysis/report.py` (writes `summary.md` + tables + figs)
- [X] T407 Implement Hydra entrypoint in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/cli/compare_runs.py` (validates inputs and writes `run_meta.json`)

## User Story 5 (P2): Generate Vidur profiling bundle

**Goal**: Generate and stage Vidur profiling inputs under an explicit profiling root, with metadata.

**Independent Test**: Run profiling on A100 and verify required files and recorded identifiers exist under the profiling root.

### Validation (US5)

- [X] T501 [P] Manual smoke run script in `/data1/huangzhe/code/gpu-simulate-test/tests/manual/test_vidur_profile_smoke.py` (A100-required; writes to `tmp/vidur_profiling/`)

### Implementation (US5)

- [X] T502 Implement profiling runner wrapper in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_ext/profile_runner.py` (calls `vidur.profiling.*` modules; stages outputs)
- [X] T503 Implement Hydra entrypoint in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/cli/vidur_profile.py` (writes metadata + fail-fast on missing GPU/model reference)

## User Story 6 (P2): Run Vidur without patching the submodule

**Goal**: Ensure Vidur recognizes `Qwen/Qwen3-0.6B` via in-repo adapters/wrappers only.

**Independent Test**: Run `vidur-profile` or `vidur-sim` using the model id and verify no modifications under `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur/`.

### Validation (US6)

- [X] T601 [P] Manual verification step: run `git status` and confirm no diffs under `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur/` after a full workflow run (document in `tmp/.../summary.md`)

### Implementation (US6)

- [X] T602 Implement Qwen3 model config registration in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_ext/qwen3_model_config.py`
- [X] T603 Implement Vidur wrapper that imports registrations before invoking Vidur in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_ext/run_vidur.py`
- [X] T604 Ensure `vidur-profile` and `vidur-sim` entrypoints use the wrapper codepaths and do not require any changes in the Vidur submodule

## Phase N: Documentation + polish

- [X] T901 Update `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/quickstart.md` with the exact `pixi run ...` commands after implementation lands
- [X] T902 Add/refresh runbook notes under `/data1/huangzhe/code/gpu-simulate-test/context/` for common failure modes (missing GPU, missing profiling data, tokenization mismatch)
