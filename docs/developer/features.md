# Features (from `tasks.md`)

This page maps the implemented feature sets to concrete entrypoints, modules, and validation.

Two task checklists exist:

- `specs/001-compare-vidur-real-timing/tasks.md`
- `specs/002-reproduce-vidur-paper-fidelity/tasks.md`

## Phase 1: Shared foundation (Hydra + artifacts)

- Hydra config tree: `configs/compare_vidur_real/`
- Pixi tasks (user-facing commands): `pyproject.toml` → `[tool.pixi.tasks]`
- Shared config + resolvers: `src/gpu_simulate_test/config/`
- Artifact IO helpers + stable ids: `src/gpu_simulate_test/io/`
- Unit validation: `tests/unit/test_artifact_schemas.py`

## User Story 1: Deterministic workload spec

- Entrypoint: `pixi run workload-spec` → `src/gpu_simulate_test/cli/workload_spec.py`
- Modules:
  - prompts IO: `src/gpu_simulate_test/workloads/prompts.py`
  - token counting: `src/gpu_simulate_test/workloads/token_lengths.py`
  - arrival schedule: `src/gpu_simulate_test/workloads/arrival_schedule.py`
- Validation:
  - unit: `tests/unit/test_workload_spec_determinism.py`
  - manual: `tests/manual/test_workload_spec_smoke.py`

## User Story 2: Real A100 timing (transformers + sarathi)

- Entrypoint: `pixi run real-bench` → `src/gpu_simulate_test/cli/real_bench.py`
- Modules:
  - trace replay: `src/gpu_simulate_test/real_bench/replay.py`
  - backends:
    - transformers: `src/gpu_simulate_test/real_bench/backends/transformers_backend.py`
    - sarathi: `src/gpu_simulate_test/real_bench/backends/sarathi_backend.py`
  - standardized metrics writer: `src/gpu_simulate_test/real_bench/metrics.py`
- Validation:
  - unit: `tests/unit/test_real_metrics_schema.py`
  - manual (GPU): `tests/manual/test_real_bench_smoke.py`

## User Story 3: Vidur simulation from repo root

- Entrypoint: `pixi run vidur-sim` → `src/gpu_simulate_test/cli/vidur_sim.py`
- Modules:
  - profiling root layout + validation: `src/gpu_simulate_test/vidur_ext/profiling_root.py`
  - sim runner wrapper + standardization: `src/gpu_simulate_test/vidur_ext/sim_runner.py`
- Validation:
  - unit prereqs: `tests/unit/test_vidur_sim_prereqs.py`
  - manual (needs profiling bundle): `tests/manual/test_vidur_sim_smoke.py`

## User Story 4: Comparison report (real vs sim)

- Entrypoint: `pixi run compare-runs` → `src/gpu_simulate_test/cli/compare_runs.py`
- Modules:
  - load metrics: `src/gpu_simulate_test/analysis/load_metrics.py`
  - token alignment + percentiles: `src/gpu_simulate_test/analysis/compare.py`
  - plotting: `src/gpu_simulate_test/analysis/plots.py`
  - report writer: `src/gpu_simulate_test/analysis/report.py`
- Validation:
  - unit alignment: `tests/unit/test_compare_alignment.py`
  - manual: `tests/manual/test_compare_runs_smoke.py`

## User Story 5: Vidur profiling bundle

- Entrypoint: `pixi run vidur-profile` → `src/gpu_simulate_test/cli/vidur_profile.py`
- Modules:
  - profiling runner wrapper: `src/gpu_simulate_test/vidur_ext/profile_runner.py`
- Validation:
  - manual (GPU): `tests/manual/test_vidur_profile_smoke.py`

## User Story 6: Qwen3 support without patching Vidur

- In-repo Vidur registration (no Vidur submodule edits required):
  - model config registration: `src/gpu_simulate_test/vidur_ext/qwen3_model_config.py`
  - wrapper that imports registration before running Vidur: `src/gpu_simulate_test/vidur_ext/run_vidur.py`

## Documentation + runbooks

- Quickstart commands: `specs/001-compare-vidur-real-timing/quickstart.md`
- Troubleshooting: `context/runbooks/001-compare-vidur-real-timing-troubleshooting.md`

---

## Paper fidelity (`002-reproduce-vidur-paper-fidelity`)

### Entrypoints

- Pixi task: `pixi run paper-fidelity` → `src/gpu_simulate_test/cli/paper_fidelity.py`
- Hydra configs: `configs/paper_fidelity/` (stage configs + `scenario/` and `workload/` groups)

### Core modules

- Trace schema + conversions + subsetting:
  - `src/gpu_simulate_test/paper_fidelity/traces.py` (canonical `trace.csv`, validators, Poisson arrivals, `apply_trace_subset`)
- Stable artifact paths + provenance helpers:
  - `src/gpu_simulate_test/paper_fidelity/paths.py`
- Vidur simulation (paper metric columns):
  - `src/gpu_simulate_test/vidur_ext/sim_runner.py` (`run_vidur_paper_fidelity_sim`)
- Sarathi real replay (paper metric columns):
  - `src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py`
- Dynamic capacity search (85% operating point):
  - `src/gpu_simulate_test/paper_fidelity/capacity.py`
- Scoring + report:
  - `src/gpu_simulate_test/paper_fidelity/scoring.py`
  - `src/gpu_simulate_test/paper_fidelity/report.py`
- Host profiling workflow:
  - `src/gpu_simulate_test/paper_fidelity/profiling.py`
  - `src/gpu_simulate_test/vidur_ext/profile_runner.py`

### Validation

- Unit tests:
  - `tests/unit/test_paper_fidelity_trace.py`
  - `tests/unit/test_paper_fidelity_capacity_search.py`
  - `tests/unit/test_paper_fidelity_profiling_root.py`
  - `tests/test_paper_fidelity_scorer.py`
- Manual smoke scripts:
  - `tests/manual/test_paper_fidelity_trace_smoke.py`
  - `tests/manual/test_paper_fidelity_vidur_sim_smoke.py`
  - `tests/manual/test_paper_fidelity_real_smoke.py` (GPU)
  - `tests/manual/test_paper_fidelity_repro_smoke.py` (GPU)

### Docs

- Quickstart (paper fidelity): `specs/002-reproduce-vidur-paper-fidelity/quickstart.md`
