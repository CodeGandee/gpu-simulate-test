# Q&A: 001-compare-vidur-real-timing

## Introduction

This Q&A doc captures implementation questions and answers for the `001-compare-vidur-real-timing` workflow, intended for developers (including future maintainers).

**Related docs**
- `context/tasks/working/001-compare-vidur-real-timing/impl-integrate-phases.md`
- `context/tasks/working/001-compare-vidur-real-timing/impl-phase-1-foundation.md`
- `context/tasks/working/001-compare-vidur-real-timing/impl-phase-2-workload-spec.md`
- `context/tasks/working/001-compare-vidur-real-timing/impl-phase-3-real-bench.md`
- `context/tasks/working/001-compare-vidur-real-timing/impl-phase-4-vidur-integration.md`
- `context/tasks/working/001-compare-vidur-real-timing/impl-phase-5-compare-runs.md`
- `context/tasks/working/001-compare-vidur-real-timing/impl-phase-6-vidur-profile.md`
- `context/tasks/working/001-compare-vidur-real-timing/impl-phase-7-docs.md`
- `specs/001-compare-vidur-real-timing/spec.md`
- `specs/001-compare-vidur-real-timing/plan.md`
- `specs/001-compare-vidur-real-timing/tasks.md`
- `specs/001-compare-vidur-real-timing/quickstart.md`
- `context/runbooks/001-compare-vidur-real-timing-troubleshooting.md`

**Key entrypoints and modules**
- `pyproject.toml`
- `configs/compare_vidur_real/workload_spec.yaml`
- `configs/compare_vidur_real/real_bench.yaml`
- `configs/compare_vidur_real/vidur_profile.yaml`
- `configs/compare_vidur_real/vidur_sim.yaml`
- `configs/compare_vidur_real/compare_runs.yaml`
- `src/gpu_simulate_test/cli/workload_spec.py`
- `src/gpu_simulate_test/cli/real_bench.py`
- `src/gpu_simulate_test/cli/vidur_profile.py`
- `src/gpu_simulate_test/cli/vidur_sim.py`
- `src/gpu_simulate_test/cli/compare_runs.py`
- `src/gpu_simulate_test/config/resolvers.py`
- `src/gpu_simulate_test/vidur_ext/profile_runner.py`
- `src/gpu_simulate_test/vidur_ext/sim_runner.py`

## How do I run the Qwen3-0.6B model using Vidur?
> Last revised at: `2025-12-31T04:15:29Z` | Last revised base commit: `4d3b6c860c5860e02595f303318841cb5ec616db`

- Ensure the model reference exists (needs `models/qwen3-0.6b/source-data/config.json`): `bash models/qwen3-0.6b/bootstrap.sh` (see `models/qwen3-0.6b/README.md`).
- Generate Vidur profiling data (GPU-required): `pixi run vidur-profile model=qwen3_0_6b hardware=a100 vidur.profiling.root=tmp/vidur_profiling/a100/qwen3_0_6b` (see `src/gpu_simulate_test/cli/vidur_profile.py`).
- Generate (or reuse) a workload dir: `pixi run workload-spec model=qwen3_0_6b ...` which writes `tmp/workloads/<workload_id>/trace_lengths.csv` and `tmp/workloads/<workload_id>/trace_intervals.csv` (see `src/gpu_simulate_test/cli/workload_spec.py`).
- Run the simulation: `pixi run vidur-sim model=qwen3_0_6b hardware=a100 vidur.profiling.root=tmp/vidur_profiling/a100/qwen3_0_6b workload.workload_dir=tmp/workloads/<workload_id>` (see `specs/001-compare-vidur-real-timing/quickstart.md`).
- Read results under `tmp/vidur_runs/<run_id>/` (standardized `request_metrics.csv`, `token_metrics.csv`, plus `run_meta.json`) written by `src/gpu_simulate_test/vidur_ext/sim_runner.py`.
- Under the hood, `model=qwen3_0_6b` maps to `model_id: Qwen/Qwen3-0.6B` (`configs/compare_vidur_real/model/qwen3_0_6b.yaml`), and `src/gpu_simulate_test/cli/vidur_sim.py` calls `register_qwen3_0_6b()` to make Vidur resolve the model via subclass discovery (`src/gpu_simulate_test/vidur_ext/qwen3_model_config.py`).

## TBD
> Last revised at: `2025-12-31T04:04:43Z` | Last revised base commit: `96913e1381d73255505461738ee8f13d25f80878`

TBD

## TBD
> Last revised at: `2025-12-31T04:04:43Z` | Last revised base commit: `96913e1381d73255505461738ee8f13d25f80878`

TBD

## TBD
> Last revised at: `2025-12-31T04:04:43Z` | Last revised base commit: `96913e1381d73255505461738ee8f13d25f80878`

TBD

## TBD
> Last revised at: `2025-12-31T04:04:43Z` | Last revised base commit: `96913e1381d73255505461738ee8f13d25f80878`

TBD

## TBD
> Last revised at: `2025-12-31T04:04:43Z` | Last revised base commit: `96913e1381d73255505461738ee8f13d25f80878`

TBD
