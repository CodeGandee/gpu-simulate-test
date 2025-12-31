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
> Last revised at: `2025-12-31T07:07:01Z` | Last revised base commit: `15808a6275e3927800856d66ebcba4240380f3f4`

- Ensure the model reference exists (needs `models/qwen3-0.6b/source-data/config.json`): `bash models/qwen3-0.6b/bootstrap.sh` (see `models/qwen3-0.6b/README.md`).
- Generate Vidur profiling data (GPU-required): `pixi run vidur-profile model=qwen3_0_6b hardware=a100 vidur.profiling.root=tmp/vidur_profiling/a100/qwen3_0_6b` (see `src/gpu_simulate_test/cli/vidur_profile.py`).
- Generate (or reuse) a workload dir: `pixi run workload-spec model=qwen3_0_6b ...` which writes `tmp/workloads/<workload_id>/trace_lengths.csv` and `tmp/workloads/<workload_id>/trace_intervals.csv` (see `src/gpu_simulate_test/cli/workload_spec.py`).
- Run the simulation: `pixi run vidur-sim model=qwen3_0_6b hardware=a100 vidur.profiling.root=tmp/vidur_profiling/a100/qwen3_0_6b workload.workload_dir=tmp/workloads/<workload_id>` (see `specs/001-compare-vidur-real-timing/quickstart.md`).
- Read results under `tmp/vidur_runs/<run_id>/` (standardized `request_metrics.csv`, `token_metrics.csv`, plus `run_meta.json`) written by `src/gpu_simulate_test/vidur_ext/sim_runner.py`.
- The `vidur-sim` outputs are **CPU-side simulation** results (predicted from profiling data), not real end-to-end GPU timing; for actual GPU timings use `pixi run real-bench ...` and compare via `pixi run compare-runs ...`.
- Under the hood, `model=qwen3_0_6b` maps to `model_id: Qwen/Qwen3-0.6B` (`configs/compare_vidur_real/model/qwen3_0_6b.yaml`), and `src/gpu_simulate_test/cli/vidur_sim.py` calls `register_qwen3_0_6b()` to make Vidur resolve the model via subclass discovery (`src/gpu_simulate_test/vidur_ext/qwen3_model_config.py`).

## How do I run the Qwen3-0.6B model using Sarathi-Serve to get real GPU timings comparable to Vidur simulation?
> Last revised at: `2025-12-31T08:13:58Z` | Last revised base commit: `947b52d8d01243d4ec5b6fefd09072b0102f50ae`

- Ensure Sarathi-Serve is available in the Pixi env: `git submodule update --init --recursive && pixi install` (Sarathi is an editable dep in `pyproject.toml` via `extern/tracked/sarathi-serve`).
- Ensure Qwen3 weights/config are present locally (recommended): `bash models/qwen3-0.6b/bootstrap.sh` so `models/qwen3-0.6b/source-data/` points at your local model directory (see `models/qwen3-0.6b/README.md`).
- Generate (or reuse) a workload spec to replay: `pixi run workload-spec model=qwen3_0_6b ...` which writes `tmp/workloads/<workload_id>/` (see `src/gpu_simulate_test/cli/workload_spec.py`).
- Run the real benchmark using Sarathi: `CUDA_VISIBLE_DEVICES=0 pixi run real-bench backend=sarathi model=qwen3_0_6b workload.workload_dir=tmp/workloads/<workload_id>` (entrypoint: `src/gpu_simulate_test/cli/real_bench.py`).
- If you want Sarathi to load from the local `models/qwen3-0.6b/source-data/` directory instead of the HuggingFace hub id, override: `model.model_id=$(pwd)/models/qwen3-0.6b/source-data` (run from repo root; Sarathi receives `cfg.model.model_id` in `src/gpu_simulate_test/real_bench/backends/sarathi_backend.py`).
- Outputs are written under `tmp/real_runs/<run_id>/` as standardized `request_metrics.csv`, `token_metrics.csv`, and `run_meta.json` (plus Sarathi runtime artifacts under `tmp/real_runs/<run_id>/sarathi/`).
- These are **actual GPU inference** wall-clock timings (tokens timestamped after each `LLMEngine.step()`); to compare against Vidur’s **CPU-side simulation** for the same workload, run `pixi run compare-runs real_run_dir=tmp/real_runs/<run_id> sim_run_dir=tmp/vidur_runs/<run_id>`.
