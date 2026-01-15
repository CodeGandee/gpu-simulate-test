# About Vidur CPU Overhead Profiling (Sarathi-Serve timers)

This note explains what Vidur’s “CPU overhead profiling” measures, how it is produced, what metrics you get, and why it can be unstable run-to-run on real machines.

Sources / upstream context:

- Vidur (simulator): https://github.com/microsoft/vidur
- Sarathi-Serve (real backend used for timing + CPU timers): https://github.com/microsoft/sarathi-serve
- Sarathi-Serve paper: https://arxiv.org/abs/2403.02310

Local code pointers (this repo):

- Vidur CPU overhead benchmark runner: `extern/tracked/vidur/vidur/profiling/cpu_overhead/benchmark_runner.py`
- Sarathi CPU timers + where “process_model_outputs” is timed:
  - `extern/tracked/sarathi-serve/sarathi/metrics/cpu_timer.py`
  - `extern/tracked/sarathi-serve/sarathi/engine/base_llm_engine.py`
- `vidur-cli` entrypoint: `src/gpu_simulate_test/cli/vidur_cli.py`
- `vidur-cli` profiling stage: `src/gpu_simulate_test/vidur_cli/stages.py`
- Pinning/guardrails for CUDA visibility and Sarathi behavior: `src/gpu_simulate_test/env_guard.py`

## 1. What “CPU overhead profiling” is for

Vidur’s core compute profiling (MLP/attention) gives you “GPU-side” kernel timing models.

But real serving has additional overhead from the host runtime and orchestration (scheduler decisions, request bookkeeping, sampler glue, etc.). Vidur’s CPU overhead profiling tries to capture those costs and provide a per-batch-size model that the simulator can add on top of GPU compute.

In practice: if the CPU overhead profile is missing or inaccurate, Vidur sim-vs-real comparisons can drift significantly, and the drift can look like a “global multiplier” error (prefill and decode both off in the same direction).

## 2. What is measured (the metrics in `cpu_overheads.csv`)

The CPU overhead profiler reports a CSV with per-batch-size rows, including both means and medians.

These values are **derived from Sarathi-Serve’s internal timers** (`CpuOperationMetrics`) collected during repeated `LLMEngine.step()` iterations in a benchmark loop.

Important: Sarathi’s `CpuTimer` calls `torch.cuda.synchronize()` when the timer exits (`extern/tracked/sarathi-serve/sarathi/metrics/cpu_timer.py`). That means these “CPU timers” can include GPU synchronization/wait time depending on what happens around the timer boundary; they are best interpreted as “host-side wall-time segments in the real stack,” not pure CPU instruction time.

### 2.1 Timer-backed metrics (from Sarathi)

The profiler records at least the following (names match CSV columns):

- `schedule_mean` / `schedule_median`
  - Time spent in `scheduler.schedule()` (wrapped by Sarathi’s CPU timer).
- `prepare_inputs_e2e_mean` / `prepare_inputs_e2e_median`
  - Worker-side time to materialize model inputs for a step (token/position tensors, padding, etc.).
- `model_execution_e2e_mean` / `model_execution_e2e_median`
  - Worker-side end-to-end model forward execution time for a step.
- `sampler_e2e_mean` / `sampler_e2e_median`
  - Worker-side time spent in sampling / logits post-processing.
- `process_model_outputs_mean` / `process_model_outputs_median`
  - Time spent applying the step outputs back into Sarathi’s sequence manager and scheduler bookkeeping (`seq_manager.on_step_completed(...)` + `scheduler.on_step_completed()`).

### 2.2 Residual metric: `ray_comm_time_mean`

`ray_comm_time_mean` is not timed directly. It is computed as a residual in Vidur’s benchmark runner:

```python
# extern/tracked/vidur/vidur/profiling/cpu_overhead/benchmark_runner.py
ray_comm_time_mean = ((end_time - start_time) - total_recorded_cpu_time) / num_steps
ray_comm_time_mean *= 1e3  # ms
```

Interpretation: “unattributed time per engine step” after subtracting the sum of recorded Sarathi timers from the benchmark’s wall time. It can include Ray/ZMQ glue, Python overhead, waiting, and any time not covered by the recorded CPU timers.

## 3. How the profiling run works (high-level)

Vidur’s CPU overhead profiler runs a Sarathi engine under a controlled synthetic workload:

1) For each tensor-parallel degree (TP) and each batch size in a predefined grid:
2) Create a Sarathi `LLMEngine` with:
   - `dtype=float16`
   - `load_format=dummy` (weights don’t need to be real for timing shape-level behavior)
   - `scheduler_config.max_num_seqs = batch_size`
   - metrics enabled, including CPU op metrics
3) Warm up:
   - add 1 request, run until it completes, reset metrics
4) Benchmark:
   - add `batch_size` requests
   - repeatedly call `engine.step()` until all requests complete
5) Read Sarathi’s metric store and emit:
   - mean and median for each CPU op timer
   - a computed `ray_comm_time_mean` residual per step

This produces a `cpu_overheads.csv` that Vidur then uses to fit predictors indexed by `(batch_size,)`.

## 4. Where outputs go (in this repo’s workflows)

In `vidur-cli`, `svr profile` writes a profiling root under the run directory:

- `<run_dir>/profile/data/profiling/cpu_overhead/a100_pairwise_nvlink/meta-llama/Llama-2-7b-hf/cpu_overheads.csv`

In sim runs, Vidur loads this file and predicts per-batch-size CPU overhead components unless CPU overhead modeling is disabled (`skip_cpu_overhead_modeling=true`).

## 5. Known problem: CPU overhead profiling can be unstable

On real machines, CPU overhead profiling can show **large run-to-run variance**, even with identical model/hardware settings and GPU pinning.

This matters because those CPU overhead values are load-bearing inputs to the simulator: different profiling roots can produce noticeably different sim-vs-real %errors.

See also: `context/issues/known/issue-vidur-cpu-overhead-profiling-is-unstable.md`.

### 5.1 Snapshot: measured instability (3-run repeatability; 2026-01-15)

We ran `vidur-cli svr profile` three times (LLaMA2-7B, A100, Sarathi, TP=1) with `GSIM_CUDA_VISIBLE_DEVICES=4,5`.

Focus `batch_size=16` (the default `backend.scheduler.max_num_seqs=16`):

| metric | run1 | run2 | run3 | mean | std | cv% |
|--------|------|------|------|------|-----|-----|
| schedule_mean | 0.220916 | 0.069194 | 0.206576 | 0.165562 | 0.083764 | 50.59% |
| sampler_e2e_mean | 0.375235 | 0.365370 | 0.369341 | 0.369982 | 0.004964 | 1.34% |
| prepare_inputs_e2e_mean | 0.125895 | 0.119867 | 0.123157 | 0.122973 | 0.003018 | 2.45% |
| model_execution_e2e_mean | 16.024888 | 15.822761 | 15.892658 | 15.913436 | 0.102653 | 0.65% |
| process_model_outputs_mean | 1.458102 | 0.499692 | 1.389881 | 1.115892 | 0.534734 | 47.92% |
| ray_comm_time_mean | 7.373335 | 3.603850 | 6.996674 | 5.991286 | 2.076140 | 34.65% |

Max variability across batch sizes (TP=1):

| metric | max cv% | batch_size@max |
|--------|---------|----------------|
| schedule_mean | 50.59% | 16 |
| sampler_e2e_mean | 10.25% | 64 |
| prepare_inputs_e2e_mean | 14.30% | 64 |
| model_execution_e2e_mean | 1.49% | 40 |
| process_model_outputs_mean | 47.92% | 16 |
| ray_comm_time_mean | 36.13% | 64 |

Notably, in the same experiment:

- `attention.csv` compute profiling was bit-identical across runs.
- `mlp.csv` drifted slightly.

### 5.2 Why this can happen (practical interpretation)

CPU overhead profiling is sensitive to host runtime noise because it is effectively measuring end-to-end serving-stack behavior (not a pure GPU kernel microbenchmark).

Likely contributors:

- OS scheduling jitter / background CPU load
- Ray/ZMQ overhead fluctuations (especially visible in the residual `ray_comm_time_mean`)
- Too-short effective measurement windows for some batch sizes (not enough steps)
- Warmup that doesn’t match steady-state at the target batch size (warmup uses 1 request, then measurement uses `batch_size` requests)

### 5.3 Mitigations / knobs to consider

If you need more stable CPU overhead profiles:

- Run multiple profiling trials and aggregate (median / trimmed mean).
- Increase warmup (ideally warm up at the same batch size as measurement).
- Increase the number of measured steps (e.g., longer decode lengths per request) to reduce noise.
- Stabilize CPU execution environment (thread caps, CPU affinity, minimize background load).
- Consider profiling with a single pinned GPU (even on multi-GPU machines) to reduce resource ambiguity.

