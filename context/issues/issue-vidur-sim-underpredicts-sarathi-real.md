# Issue: Vidur Simulation Underpredicts Sarathi Real Latency for LLaMA2-7B

**Status**: Confirmed
**Date**: 2026-01-09
**Priority**: High (Fidelity Gap)

## Observation

In paper-fidelity experiments for LLaMA2-7B (TP=1) on A100, Vidur simulation consistently underpredicts real-world (Sarathi-Serve) latency by approximately **25%**.

**Example Report**: `results/reports/2026-01-08/paper_fidelity/llama2_7b_arxiv_sim_vs_real_2026-01-08_12-22-51-887166845_static_medium`

| Metric | Sim (P50) | Real (P50) | % Error |
| :--- | :--- | :--- | :--- |
| `request_execution_plus_preemption_time_normalized` | 0.027 | 0.036 | **24.58%** |
| `request_e2e_time_normalized` | 3.00 | 3.94 | **23.89%** |

The error is consistent across percentiles (P50/P95) and metrics.

## Root Cause Analysis

The primary cause is that **CPU overhead modeling is disabled in the Vidur sim path** (`scenario.vidur.enable_cpu_overhead_modeling=false`, which maps to Vidur’s internal `skip_cpu_overhead_modeling=true`), while the real execution (Sarathi-Serve) incurs non-negligible CPU overheads (Python runtime, scheduler, input/output processing). This effect is magnified for small models like LLaMA2-7B where GPU compute times are short.

### 1. Vidur Configuration
The Vidur simulation was run with CPU overhead modeling disabled (default).
- Repo config: `scenario.vidur.enable_cpu_overhead_modeling=false`.
- Vidur internal config: `skip_cpu_overhead_modeling=true` (see `extern/tracked/vidur/vidur/config/config.py`).
- Run Metadata: `run_meta.json` confirms CPU overhead modeling is disabled (older runs may record `skip_cpu_overhead_modeling: true`; newer runs record `enable_cpu_overhead_modeling: false` in the resolved scenario config).

### 2. Vidur Implementation
When Vidur internal `skip_cpu_overhead_modeling` is `True`, the `SklearnExecutionTimePredictor` explicitly zeroes out:
- `schedule`
- `sampler_e2e`
- `prepare_inputs_e2e`
- `process_model_outputs`
- `ray_comm_time`

See `extern/tracked/vidur/vidur/execution_time_predictor/sklearn_execution_time_predictor.py`.

### 3. Missing Profiling Data
The profiling bundle used (`results/raw/vidur-profiling/llama2-7b/sarathi-serve/...`) **does not contain CPU overhead data**.
- Validated via file inspection: `data/profiling/cpu_overhead/` is missing.
- `src/gpu_simulate_test/vidur_ext/profile_runner.py` defaults `include_cpu_overhead=False` to match the Vidur paper's methodology.

### 4. Paper Reference
The Vidur paper (`extern/tracked/vidur/paper/tex/5-eval.tex`) explicitly acknowledges this limitation:
> "Note that we observe slightly higher average error rates for the 7B model, we attribute this to the higher CPU overhead for smaller models."

They reported up to **12.65%** error for LLaMA-7B. Our observed **25%** error suggests our specific Sarathi environment (or host CPU) has higher overheads than the "optimized version of the vLLM codebase" used in the paper.

## Proposed Resolution

To reduce the Sim-vs-Real gap and achieve higher fidelity for LLaMA2-7B:

1.  **Enable CPU Overhead Profiling**:
    - Update the `paper-fidelity profile` command (and underlying `profile_runner.py`) to set `include_cpu_overhead=True` when requested.
    - Generate a new profiling bundle that includes `cpu_overhead.csv`.

2.  **Enable CPU Overhead Modeling in Sim**:
    - Update `paper-fidelity repro` / `vidur-sim` to set `scenario.vidur.enable_cpu_overhead_modeling=true` when CPU overhead data is available (maps to Vidur internal `skip_cpu_overhead_modeling=false`).

3.  **Investigate Sarathi Overheads**:
    - If the gap persists >15% after modeling CPU overhead, profile the Sarathi execution itself to identify unexpected bottlenecks (e.g., inefficient tokenization, excessive logging, or unoptimized scheduler loops).

## Next Steps
- Verify if `vidur_profiling_cpu_overhead_main` works in our environment.
- Create a task to enable full-stack profiling for fidelity runs.
