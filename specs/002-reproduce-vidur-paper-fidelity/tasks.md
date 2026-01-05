# Tasks: Reproduce Vidur paper fidelity

**Input**: Design documents from `/data1/huangzhe/code/gpu-simulate-test/specs/002-reproduce-vidur-paper-fidelity/`  
**Prerequisites**: `/data1/huangzhe/code/gpu-simulate-test/specs/002-reproduce-vidur-paper-fidelity/plan.md` (required), `/data1/huangzhe/code/gpu-simulate-test/specs/002-reproduce-vidur-paper-fidelity/spec.md` (required)

**Validation**: Every change MUST include at least one of manual, unit, or integration validation under `/data1/huangzhe/code/gpu-simulate-test/tests/` (per the constitution).

**Organization**: Tasks are grouped by user story (US1–US6) to keep each story independently testable where feasible.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

- [ ] T001 [P] Create `paper_fidelity` package skeleton under `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/`
- [ ] T002 [P] Add CLI entrypoint `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/cli/paper_fidelity.py` (Hydra-driven; mirrors existing CLIs)
- [ ] T003 Add Pixi task `paper-fidelity` in `/data1/huangzhe/code/gpu-simulate-test/pyproject.toml` → `python -m gpu_simulate_test.cli.paper_fidelity`
- [ ] T004 Add Hydra configs under `/data1/huangzhe/code/gpu-simulate-test/configs/paper_fidelity/` (defaults + scenario group + baseline scenario `llama2_7b_arxiv`)

---

## Phase 2: Foundational (Blocking Prerequisites)

**⚠️ CRITICAL**: These tasks unblock all user stories by standardizing schemas, artifact locations, and provenance.

- [ ] T010 Define canonical trace schema + validator + converters in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/traces.py`
  - Accept canonical `trace.csv` (`arrived_at,num_prefill_tokens,num_decode_tokens` + optional ids)
  - Accept legacy split files (`trace_lengths.csv` + `trace_intervals.csv`) and convert deterministically
  - Support generating static traces (`arrived_at=0`) and dynamic traces (seeded Poisson arrivals)
- [ ] T011 [P] Add unit tests for trace determinism + validation errors in `/data1/huangzhe/code/gpu-simulate-test/tests/unit/test_paper_fidelity_trace.py`
- [ ] T012 [P] Add shared artifact path helpers and run metadata helpers (reuse `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/io/`) in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/paths.py`

**Checkpoint**: Trace generation/validation + reproducible artifact paths are available.

---

## Phase 3: User Story 1 - End-to-end fidelity reproduction report (Priority: P1)

**Goal**: One command produces an end-to-end `summary.md` for a baseline scenario (static + dynamic).

**Independent Test**: Run `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload static|dynamic` and verify a report exists at `/data1/huangzhe/code/gpu-simulate-test/results/reports/<date>/paper_fidelity/llama2_7b_arxiv/summary.md`.

### Validation (REQUIRED)

- [ ] T020 [P] [US1] Manual smoke run in `/data1/huangzhe/code/gpu-simulate-test/tests/manual/test_paper_fidelity_repro_smoke.py` (small trace; writes outputs under `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/`)

### Implementation

- [ ] T021 [US1] Implement `paper-fidelity repro` orchestration in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/cli/paper_fidelity.py`
  - trace → sim → (capacity + real at qps_85 for dynamic) → score → report
  - print the report directory to stdout (Hydra convention)
- [ ] T022 [US1] Implement `paper-fidelity score` (scoring-only workflow) in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/cli/paper_fidelity.py`

---

## Phase 4: User Story 2 - Standardized trace generation and validation (Priority: P1)

**Goal**: One canonical trace schema drives both simulation and real execution; legacy trace layouts still work.

**Independent Test**: Generate a trace from a token-length distribution and confirm schema validity + determinism under a fixed seed.

### Validation (REQUIRED)

- [ ] T030 [P] [US2] Manual trace generation/validation script in `/data1/huangzhe/code/gpu-simulate-test/tests/manual/test_paper_fidelity_trace_smoke.py`

### Implementation

- [ ] T031 [US2] Add `paper-fidelity trace` subcommand (generate/validate) in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/cli/paper_fidelity.py`
- [ ] T032 [US2] Wire baseline token-length source `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur/data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv` into the baseline scenario config (`/data1/huangzhe/code/gpu-simulate-test/configs/paper_fidelity/scenarios/llama2_7b_arxiv.yaml`)

---

## Phase 5: User Story 3 - Simulation run with paper-required metrics (Priority: P1)

**Goal**: Vidur simulation preserves paper-required normalized metrics without reinterpretation.

**Independent Test**: Run sim for a small trace and verify output contains normalized metric columns and a run manifest.

### Validation (REQUIRED)

- [ ] T040 [P] [US3] Manual smoke run in `/data1/huangzhe/code/gpu-simulate-test/tests/manual/test_paper_fidelity_vidur_sim_smoke.py`
- [ ] T041 [P] [US3] Unit test for schema preservation using fixtures in `/data1/huangzhe/code/gpu-simulate-test/tests/unit/test_paper_fidelity_vidur_metrics_schema.py`

### Implementation

- [ ] T042 [US3] Extend `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_ext/sim_runner.py` to:
  - accept canonical `/tmp/paper_fidelity/.../trace.csv` directly (no forced legacy conversion)
  - write a paper-fidelity `request_metrics.csv` that preserves Vidur’s normalized metric columns (e.g., `request_e2e_time_normalized`, `request_execution_plus_preemption_time_normalized`, `request_scheduling_delay`)
  - keep the raw Vidur output dir for debugging under the run directory

---

## Phase 6: User Story 4 - Real execution replay with aligned timing boundaries (Priority: P2)

**Goal**: Replay the same trace with Sarathi-Serve under load and emit request lifecycle metrics sufficient for paper-aligned scoring.

**Independent Test**: Run a small replay and confirm the output contains `request_scheduling_delay` and normalized end-to-end metrics.

### Validation (REQUIRED)

- [ ] T050 [P] [US4] Manual smoke run in `/data1/huangzhe/code/gpu-simulate-test/tests/manual/test_paper_fidelity_real_smoke.py`
- [ ] T051 [P] [US4] Unit test for converting Sarathi outputs to the paper-fidelity request_metrics schema in `/data1/huangzhe/code/gpu-simulate-test/tests/unit/test_paper_fidelity_real_metrics_schema.py` (use a checked-in fixture file)

### Implementation

- [ ] T052 [US4] Implement a Sarathi-backed trace replay runner that uses `prompt_token_ids` (token-length-only) and Sarathi’s metrics store, writing a paper-fidelity `request_metrics.csv` under `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/<scenario>/real/`
  - Prefer adding a new backend module under `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/real_bench/backends/` to avoid breaking existing `real_bench`
  - Fail fast with actionable errors when CUDA/model assets are unavailable

---

## Phase 7: User Story 5 - Capacity discovery for 85% utilization runs (Priority: P2)

**Goal**: Determine capacity QPS and compute the 85% operating point using P99 scheduling delay overload criterion.

**Independent Test**: Run a bounded search and verify it outputs `capacity_qps` and `qps_85` plus the criterion used.

### Validation (REQUIRED)

- [ ] T060 [P] [US5] Unit test for capacity search logic (pure function) in `/data1/huangzhe/code/gpu-simulate-test/tests/unit/test_paper_fidelity_capacity_search.py`
- [ ] T061 [P] [US5] Manual smoke run in `/data1/huangzhe/code/gpu-simulate-test/tests/manual/test_paper_fidelity_capacity_smoke.py` (small run; bounded range)

### Implementation

- [ ] T062 [US5] Implement capacity discovery in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/capacity.py`
  - binary search over QPS (configurable bounds/iters)
  - overload if P99(`request_scheduling_delay`) > 5s (default; configurable)
  - write `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/<scenario>/capacity/capacity.json`

---

## Phase 8: User Story 6 - Fidelity scoring and gap diagnosis (Priority: P3)

**Goal**: Compute paper-aligned percentile summaries and percent error; produce a diagnosis section when thresholds are exceeded.

**Independent Test**: Given two metrics CSVs (sim + real), scorer emits tables + pass/warn/fail + diagnosis if needed.

### Validation (REQUIRED)

- [ ] T070 [P] [US6] Unit tests with fixed fixtures in `/data1/huangzhe/code/gpu-simulate-test/tests/test_paper_fidelity_scorer.py` (include fixture CSVs under `/data1/huangzhe/code/gpu-simulate-test/tests/fixtures/paper_fidelity/`)

### Implementation

- [ ] T071 [US6] Implement scorer in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/scoring.py`
  - compute P50/P95 (minimum) for static/dynamic metrics for sim + real
  - compute percent error `abs(sim - real) / real`
  - apply thresholds (Pass ≤5%, Warn 5–9%, Fail >9% by default; configurable)
- [ ] T072 [US6] Implement report writer in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/report.py`
  - write `summary.md` plus optional tables/plots under `/data1/huangzhe/code/gpu-simulate-test/results/reports/<date>/paper_fidelity/<scenario>/`
- [ ] T073 [US6] Add “gap diagnosis” heuristics (triggered by threshold exceedance) with at least one concrete hypothesis and evidence reference in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/paper_fidelity/report.py`

---

## Dependencies & Execution Order

- **Phase 1–2** block everything else: standard schemas, configs, and artifact paths must exist first.
- **US3 + US4** must complete before meaningful scoring (US6) and end-to-end repro (US1).
- **US5** is required for dynamic runs at the 85% operating point (US1 dynamic workflow).
- **US6** is required for scoring-only (US1 acceptance scenario 2) and for producing `summary.md`.

