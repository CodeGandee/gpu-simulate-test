# Architecture

## High-level pipeline

1. `workload-spec` produces a deterministic workload directory (`tmp/workloads/<workload_id>/`).
2. `real-bench` replays the workload against a real backend and writes standardized metrics (`tmp/real_runs/<run_id>/`).
3. `vidur-profile` generates a profiling bundle (`tmp/vidur_profiling/...`) used by the simulator.
4. `vidur-sim` runs Vidur from repo root and writes standardized metrics (`tmp/vidur_runs/<run_id>/`).
5. `compare-runs` loads both runs, aligns tokens, and writes a report (`tmp/comparisons/<comparison_id>/`).

## Design constraints (from the spec)

- Deterministic workload generation (`seed` + trace files).
- All timestamps are integer nanoseconds (relative to run start, monotonic).
- Outputs default to `tmp/` (avoid committing large artifacts).
- “Early stop” must be handled by recording `num_decode_tokens_actual` and truncating simulated tokens during comparison.
- Vidur must run from repo root using an explicit profiling root (not submodule-relative paths).

## Key directories

- `configs/compare_vidur_real/`: Hydra presets (one stage config per command + groups)
- `src/gpu_simulate_test/cli/`: Hydra entrypoints (Pixi tasks call these)
- `src/gpu_simulate_test/workloads/`: workload spec generation
- `src/gpu_simulate_test/real_bench/`: real timing harness
- `src/gpu_simulate_test/vidur_ext/`: Vidur adapters/wrappers (no Vidur submodule patching required)
- `src/gpu_simulate_test/analysis/`: compare + report generation

