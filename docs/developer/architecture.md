# Architecture

## High-level pipelines

### A) Compare Vidur vs real timing (`001-compare-vidur-real-timing`)

1. `workload-spec` produces a deterministic workload directory (`tmp/workloads/<workload_id>/`).
2. `real-bench` replays the workload against a real backend and writes standardized metrics (`tmp/real_runs/<run_id>/`).
3. `vidur-profile` generates a profiling bundle (`tmp/vidur_profiling/...`) used by the simulator.
4. `vidur-sim` runs Vidur from repo root and writes standardized metrics (`tmp/vidur_runs/<run_id>/`).
5. `compare-runs` loads both runs, aligns tokens, and writes a report (`tmp/comparisons/<comparison_id>/`).

### B) Paper fidelity reproduction (`002-reproduce-vidur-paper-fidelity`)

1. `paper-fidelity trace` produces a canonical `trace.csv` (`tmp/paper_fidelity/traces/<scenario>/trace.csv`).
2. `paper-fidelity repro` orchestrates:
   - trace build (static/dynamic),
   - dynamic capacity search for the 85% operating point (dynamic only),
   - Vidur simulation (paper metric columns),
   - Sarathi real replay (paper metric columns),
   - scoring + report (`results/reports/<date>/paper_fidelity/<scenario>/summary.md`).
3. `paper-fidelity score` can run scoring only (given existing metrics CSVs).
4. `paper-fidelity profile` generates a host profiling root and can be used to run “host-calibrated” simulations.

## Design constraints (from the spec)

- Deterministic workload generation (`seed` + trace files).
- All timestamps are integer nanoseconds (relative to run start, monotonic).
- Outputs default to `tmp/` (avoid committing large artifacts).
- “Early stop” must be handled by recording `num_decode_tokens_actual` and truncating simulated tokens during comparison.
- Vidur must run from repo root using an explicit profiling root (not submodule-relative paths).

## Design choices (paper fidelity)

- One canonical trace schema (`trace.csv`) drives both sim and real replay (`arrived_at,num_prefill_tokens,num_decode_tokens`).
- Prefer in-engine timing/metrics from Sarathi (avoid client-side timestamping drift).
- Trace subsetting is by **row index**:
  - `range` is allowed for all sources (timed and untimed).
  - `indices` is only allowed for untimed sources (those where arrivals are generated inside the workflow).
  - For timed sources, `range` selection rebases `arrived_at` so the subset starts at `0.0`.
- Subsetting never mutates the source trace input; it only affects derived `trace.csv` under `tmp/paper_fidelity/`.

## Key directories

- Compare workflow (001):
  - `configs/compare_vidur_real/`: Hydra presets (one stage config per command + groups)
  - `src/gpu_simulate_test/workloads/`: workload spec generation
  - `src/gpu_simulate_test/analysis/`: compare + report generation
- Paper fidelity workflow (002):
  - `configs/paper_fidelity/`: Hydra presets (`repro.yaml`, `trace.yaml`, etc)
  - `src/gpu_simulate_test/paper_fidelity/`: trace, capacity, scoring, report, profiling
- Shared:
  - `src/gpu_simulate_test/cli/`: Hydra entrypoints (Pixi tasks call these)
  - `src/gpu_simulate_test/real_bench/`: real timing harnesses (includes Sarathi paper-fidelity backend)
  - `src/gpu_simulate_test/vidur_ext/`: Vidur adapters/wrappers (no Vidur submodule patching required)
