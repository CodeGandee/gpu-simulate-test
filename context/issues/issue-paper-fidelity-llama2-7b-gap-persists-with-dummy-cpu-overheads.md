# Issue: Paper-fidelity LLaMA2-7B gap persists with “CPU overhead modeling” enabled

**Status**: Observed
**Date**: 2026-01-09
**Priority**: High (Fidelity gap)

## Observation

Report: `results/reports/2026-01-09/paper_fidelity/llama2_7b_arxiv/summary.md`

Even when the Vidur simulation is configured to include CPU overhead (Vidur internal `skip_cpu_overhead_modeling=false`), the simulation still **underpredicts** Sarathi real by ~**16–20%** across both request-level and stage-level metrics.

From the report:

| Metric | Percentile | Sim | Real | Percent error |
| :--- | :--- | :--- | :--- | :--- |
| `request_execution_plus_preemption_time_normalized` | p50 | 0.0286743 | 0.035372 | 18.94% |
| `request_execution_plus_preemption_time_normalized` | p95 | 0.0590226 | 0.0701305 | 15.84% |
| `request_e2e_time_normalized` | p50 | 0.374385 | 0.451723 | 17.12% |
| `request_e2e_time_normalized` | p95 | 0.946374 | 1.14568 | 17.40% |
| `prefill_time_execution_plus_preemption_normalized` | p50 | 0.000916993 | 0.00111189 | 17.53% |
| `decode_time_execution_plus_preemption_normalized` | p50 | 0.0137182 | 0.0169171 | 18.91% |

Notably, prefill and decode have **similar** percent error (no obvious “prefill too fast / decode too slow” skew), suggesting this gap is closer to a **global underprediction** than a scheduler-knob mismatch.

## Key finding: the CPU-overhead data used is dummy / unprofiled

This report uses a custom profiling root:

- Profiling root: `tmp/test_profiling_root` (see report “Profiling” section)

Within that root, CPU overheads are present at:

- `tmp/test_profiling_root/data/profiling/cpu_overhead/a100_pairwise_nvlink/meta-llama/Llama-2-7b-hf/cpu_overheads.csv`

However, the file contents are clearly synthetic:

- Every overhead column is constant across all batch sizes (128 rows): e.g. `schedule_median` is always `0.1`, `ray_comm_time_mean` always `0.5`, `model_execution_e2e_median` always `10.0`.
- This matches exactly the output of `tests/manual/generate_dummy_cpu_overhead.py`.
- `tmp/test_profiling_root/profiling_meta.json` also reports `cpu_overhead_profiled=false` and `cpu_overheads_csv=null` under `profiling_outputs`.

So, while the Vidur sim run was configured with CPU overhead modeling enabled, it was enabled against **placeholder** CPU overhead inputs, not a CPU-overhead profile measured from our real stack. The remaining ~18% underprediction is therefore not surprising.

## Likely reasons (ranked)

### 1) CPU overhead profiling did not successfully produce real `cpu_overheads.csv`

We do not currently have a host-profiled `cpu_overheads.csv` under:

- `tmp/paper_fidelity/profiling_roots/llama2_7b_arxiv/<run_id>/data/profiling/cpu_overhead/.../cpu_overheads.csv`

and thus the report used a custom “test” profiling root with dummy CPU overheads.

What we see instead is that the CPU overhead profiler produced an *empty* `cpu_overhead.csv` (no rows/columns), and our profiling pipeline never staged a real `cpu_overheads.csv` into the profiling root.

Evidence from the host profiling run outputs:

- The CPU overhead profiler output exists but is 1 byte (effectively empty):
  - `tmp/paper_fidelity/profiling_outputs/llama2_7b_arxiv/2026-01-09_06-26-32-853654/cpu_overhead/2026-01-09_06-28-24/meta-llama/Llama-2-7b-hf/cpu_overhead.csv`
- Ray logs for that timestamp show Torch CUDA initialization failing with an invalid device index:
  - `/tmp/ray/session_2026-01-09_06-28-24_940699_1787331/logs/*` contains `device=7, num_gpus=7` errors.
- On this host, Torch CUDA init fails when enumerating all GPUs (but works when restricting to a single GPU):
  - `CUDA_VISIBLE_DEVICES=0 pixi run python -c "import torch; print(torch.cuda.device_count())"` succeeds.
  - `pixi run python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"` can fail due to the same device-7 error when all GPUs are visible.

Hypothesis: one GPU is in a “bad” state for CUDA initialization (e.g., `nvidia-smi` shows GPU 7 with `GPU-Util: N/A` and `MIG M.: Enabled`), and Torch’s capability check crashes when that GPU is visible. This prevents Sarathi workers (used by Vidur’s CPU overhead profiler) from starting, so the profiler records zero rows.

### 2) Even with real CPU overheads, Vidur’s CPU overhead model may still miss Sarathi wall-clock costs

Vidur’s “CPU overhead” captures specific engine sections (schedule, sampler, prepare inputs, process outputs, etc.), but Sarathi-Serve’s end-to-end cost can include additional overhead:

- Python/Ray scheduling, synchronization, and runtime jitter
- extra framework glue not represented in Vidur’s CPU metrics
- different batching/iteration semantics that change how per-step overhead accumulates

This can leave a residual underprediction even after real CPU overhead profiling works.

### 3) Compute profiling may be optimistic vs the real kernel path

Even with host profiling, microbenchmarks can be systematically faster than real inference (e.g., different kernel path, cache effects, allocator/launch overheads). If Sarathi’s actual attention backend / dtype / kernel choices differ from what was profiled, a nearly uniform ~15–20% underprediction is plausible.

## Possible solutions / mitigations

### A) Make CPU overhead profiling actually work end-to-end

1) Ensure the CPU overhead profiler produces a *non-empty* `cpu_overhead.csv`:
   - Work around broken GPUs by setting `CUDA_VISIBLE_DEVICES` to a healthy subset before running:
     - e.g. `CUDA_VISIBLE_DEVICES=0 pixi run paper-fidelity profile --scenario llama2_7b_arxiv --include-cpu-overhead`
   - Or fix the underlying GPU state (e.g., disable MIG on the problematic GPU or make it available via MIG instances).
2) Update our CPU overhead profiling wrapper to fail fast if it produces an empty file:
   - `src/gpu_simulate_test/vidur_ext/vidur_profiling_cpu_overhead_main.py` should exit non-zero when `results` is empty.
   - `src/gpu_simulate_test/vidur_ext/profile_runner.py` should treat empty `cpu_overhead.csv` as a clear error with actionable guidance (rather than failing later with `pandas.errors.EmptyDataError`).
3) Re-run scoring using a real profiling root (not `tmp/test_profiling_root`) with:
   - sim: `scenario.vidur.skip_cpu_overhead_modeling=false`
   - profiling root containing a real `cpu_overheads.csv`

### B) Add guardrails to prevent accidental “CPU overhead enabled with dummy data”

- If `scenario.vidur.skip_cpu_overhead_modeling=false`, validate that `cpu_overheads.csv`:
  - exists, and
  - is non-degenerate (e.g., more than 1 unique value for key columns), and
  - is recorded as `cpu_overhead_profiled=true` in the profiling meta (when available).
- If validation fails, fail fast or annotate the report as “CPU overhead inputs are placeholder/untrusted”.

### C) If a residual ~10–20% gap remains after real CPU overhead profiling

- Record real-side engine config (attention backend, dtype, cache settings, CUDA graphs, etc.) into the real run provenance so we can verify profiling/real parity.
- Consider calibrating a small, explicit “unmodeled overhead” term (or a global scaling factor) for this specific stack, and document that the result is *stack-calibrated* rather than paper-reproduction.
