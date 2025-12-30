# Tasks: Compare Vidur vs real Qwen3 A100 timing

**Feature**: `001-compare-vidur-real-timing`  
**Date**: 2025-12-30  
**Spec**: `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/spec.md`  
**Plan**: `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/plan.md`

## Phase 0: Repo + environment readiness

- [ ] Confirm submodules are initialized: `git submodule update --init --recursive`
- [ ] Confirm Pixi env is usable: `pixi install` and `pixi run python -c "import torch; import vidur; import sarathi"`
- [ ] Confirm Qwen3 external reference exists: `/data1/huangzhe/code/gpu-simulate-test/models/qwen3-0.6b/source-data/`

## Phase 1: Canonical workload spec (`workload-spec`)

- [ ] Add `src/gpu_simulate_test/workloads/` with:
  - prompts loader/writer (`prompts.jsonl`)
  - deterministic trace generation (`trace_lengths.csv`, `trace_intervals.csv`)
  - `workload_meta.json` writer (git/env provenance)
- [ ] Add CLI entrypoint `workload-spec` (Pixi-first, writes under `tmp/` by default).
- [ ] Unit tests:
  - determinism: same seed → identical traces
  - schema: required columns exist; `arrival_time_ns` consistency with `inter_arrival_ns`

## Phase 2: Real benchmark (`real-bench`)

- [ ] Add `src/gpu_simulate_test/real_bench/` with a replay runner that:
  - reads workload spec
  - issues requests on the `trace_intervals.csv` schedule
  - records per-request TTFT and per-token timestamps
  - writes standardized `request_metrics.csv` + `token_metrics.csv` + `run_meta.json`
- [ ] Implement `--backend transformers` for Qwen3 ground-truth timing.
- [ ] Implement `--backend sarathi`:
  - if Qwen3 unsupported in Sarathi, fail fast with actionable error (and suggest `--backend transformers`)
  - keep schema identical across backends when successful
- [ ] Manual smoke script under `/data1/huangzhe/code/gpu-simulate-test/tests/manual/`:
  - tiny workload (2–5 requests)
  - validates outputs exist and timestamps are monotonic

## Phase 3: Vidur adapters (`vidur-profile`, `vidur-sim`)

- [ ] Add `src/gpu_simulate_test/vidur_ext/`:
  - local Vidur `BaseModelConfig` registration for `Qwen/Qwen3-0.6B`
  - wrapper entrypoints that run Vidur modules without modifying `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur/`
- [ ] Implement `vidur-profile`:
  - runs Vidur profiling modules for Qwen3
  - stages outputs under an explicit profiling root (machine-local) at `/data1/huangzhe/code/gpu-simulate-test/tmp/vidur_profiling/`
- [ ] Implement `vidur-sim`:
  - reads canonical workload spec
  - generates a Vidur trace-replay CSV input (seconds) from the canonical traces
  - runs Vidur from repo root with `execution_time_predictor_config_*_input_file` templates pointing at `--profiling-root`
  - postprocesses Vidur outputs into standardized `request_metrics.csv` + `token_metrics.csv` + `run_meta.json`
- [ ] Manual smoke script:
  - tiny workload
  - `vidur-profile` then `vidur-sim`
  - validates artifacts under `tmp/vidur_runs/`

## Phase 4: Comparison + reporting (`compare-runs`)

- [ ] Add `src/gpu_simulate_test/analysis/compare_vidur_vs_real.py`:
  - loads real + vidur standardized metrics
  - handles early stopping by truncating sim token series to `num_decode_tokens_actual`
  - computes percentile tables (P50/P90/P99) for TTFT and per-token latency
  - writes plots and `/data1/huangzhe/code/gpu-simulate-test/tmp/comparisons/<run_id>/summary.md`
- [ ] Add CLI entrypoint `compare-runs`.
- [ ] Unit tests for schema validation and early-stop truncation logic.

## Phase 5: Documentation

- [ ] Update `/data1/huangzhe/code/gpu-simulate-test/context/hints/` runbooks to use the repo-root wrappers and the profiling root approach.
- [ ] Ensure `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/quickstart.md` matches the implemented CLI flags and output locations.

## Definition of Done

- [ ] A contributor can run the end-to-end baseline (workload → real → vidur → compare) using only documented `pixi run ...` commands.
- [ ] Outputs are written under `/data1/huangzhe/code/gpu-simulate-test/tmp/` and include `run_meta.json` for provenance.
- [ ] At least one manual smoke and one unit validation exist under `/data1/huangzhe/code/gpu-simulate-test/tests/`.
