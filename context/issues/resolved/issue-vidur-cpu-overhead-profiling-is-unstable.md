# Issue: Vidur CPU overhead profiling is unstable (high run-to-run variance)

**Status**: Resolved (no longer tracked as a known issue)
**Date**: 2026-01-15
**Resolved**: 2026-01-20
**Last updated**: 2026-01-20
**Priority**: High (fidelity + reproducibility)

## Summary

Repeated runs of `vidur-cli svr profile` on the same host (same model/hardware) produce **materially different** CPU overhead profiling outputs (`cpu_overheads.csv`), even though:

- GPUs are pinned via `GSIM_CUDA_VISIBLE_DEVICES`
- Vidur’s profilers include warmup iterations
- Compute profiling outputs are stable (attention is bit-identical; MLP drifts slightly)

Because Vidur sim consumes these CPU overhead numbers, a `vidur-cli` sim-vs-real report (and any paper-fidelity run using a `vidur-cli` profiling root) can show noticeably different %errors depending on which profiling root is used.

## Resolution

As of 2026-01-20, we consider this resolved in the sense that it is no longer actively tracked as an ongoing repo
limitation. If CPU overhead profiling reproducibility becomes a requirement again, reopen as a new issue with fresh
measurements (Ray version, host load, and exact profiling command-line/env).

## Where this shows up

- Profiling output path (within a `vidur-cli` run dir):
  - `<run_dir>/profile/data/profiling/cpu_overhead/a100_pairwise_nvlink/meta-llama/Llama-2-7b-hf/cpu_overheads.csv`
- The relevant `vidur-cli` stage is:
  - `vidur-cli svr profile --run-dir <run_dir>`
  - CPU overhead profiling is **enabled by default**; disable with `--no-include-cpu-overhead`.

## Evidence (repeatability experiment)

We ran `svr profile` 3 times with identical presets:

- `model=llama2_7b hardware=a100 backend=sarathi workload=default vidur=default`
- `GSIM_CUDA_VISIBLE_DEVICES=4,5`

Workspace (tmp, not tracked):

- `tmp/profiling_repeatability_llama2_7b_a100_20260115T103526Z`
  - `run_dirs.txt`: the 3 run directories
  - `compare.md` / `compare.json`: summary comparison

Key finding: **CPU overhead is highly variable at the operating batch sizes we care about**.

### Snapshot: 3-run comparison (copied here; do not rely on tmp/)

Runs:

- run1: `tmp/profiling_repeatability_llama2_7b_a100_20260115T103526Z/sim_vs_real/m_llama2_7b+h_a100+b_sarathi+w_default+v_default+20260115T103610Z`
- run2: `tmp/profiling_repeatability_llama2_7b_a100_20260115T103526Z/sim_vs_real/m_llama2_7b+h_a100+b_sarathi+w_default+v_default+20260115T104341Z`
- run3: `tmp/profiling_repeatability_llama2_7b_a100_20260115T103526Z/sim_vs_real/m_llama2_7b+h_a100+b_sarathi+w_default+v_default+20260115T105114Z`

File fingerprints (sha256 prefix):

| Run | cpu_overheads.csv | mlp.csv | attention.csv | all_reduce.csv | send_recv.csv |
|-----|-------------------|--------|---------------|----------------|---------------|
| run1 | c1ded91bdbd3 | a77d41dd5bfd | ba619403d61e | 27428d884ae0 | b962137cf047 |
| run2 | b9cbada852a3 | 981aee03284e | ba619403d61e | 27428d884ae0 | b962137cf047 |
| run3 | bea72edbdd62 | 4079348824e1 | ba619403d61e | 27428d884ae0 | b962137cf047 |

CPU overhead compare (TP=1), focus `batch_size=16` (units are profiler-native; treated as milliseconds in Vidur):

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

By contrast:

- `attention.csv` was **bit-identical** across the 3 runs.
- `mlp.csv` changed slightly (small aggregate drift).

## Impact

1) **Sim-vs-real %error is not stable** across “same config, same trace” runs if the profiling root changes.
2) This can make comparisons look inconsistent:
   - A report that uses profiling root A can show “good” error (single-digit %).
   - A report that uses profiling root B (freshly profiled) can show materially larger error (10–20%+), even when all apple-to-apple config knobs match.

## Likely root cause (hypotheses)

CPU overhead profiling is inherently noisier than compute microbenchmarks here because it measures end-to-end serving stack CPU behavior, and it is sensitive to:

- **Host CPU load / scheduling jitter** during the benchmark window.
- **Ray overhead** (the profiler computes `ray_comm_time_mean` as the remainder of wall time minus recorded CPU op time, per engine step).
- **Short measurement windows** for some batch sizes (too few steps → higher variance).
- **Warmup may be insufficient** for steady-state at the target batch size (Vidur’s CPU overhead profiler warms up with one request, not with a full batch).

## Mitigations (today)

To reduce variance in practice:

- **Run CPU overhead profiling multiple times and aggregate** (median / trimmed mean) before trusting the numbers.
- **Pin to a single GPU** for profiling (even on a multi-GPU host) to reduce resource ambiguity:
  - e.g., `export GSIM_CUDA_VISIBLE_DEVICES=4`
- **Stabilize CPU execution environment**:
  - pin thread counts (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc.)
  - consider CPU affinity (`taskset`) / running on an otherwise idle host
- **Consider raising `backend.scheduler.max_num_seqs`** to a batch size where CPU-overhead results are empirically more stable on this host (tradeoff: it changes the serving regime and can affect memory).

## Proposed improvements (code)

Add knobs + provenance to make CPU overhead profiling more reproducible:

1) **Multiple trials per (batch_size, tp)** and emit aggregated statistics into `cpu_overheads.csv`.
2) **Stronger warmup**: warm up using the same batch size as the measurement (not 1 request).
3) **Longer measurement**: increase decode tokens / step count for CPU overhead benchmarking.
4) **Record profiling meta in `svr profile`** (so the run captures whether compute profiling fell back to templates, and which commands/params were used).

## Repro steps (minimal)

```bash
export GSIM_CUDA_VISIBLE_DEVICES=4,5
export GSIM_REPO_ROOT=\"$PWD\"
export GSIM_VIDUR_WORKSPACE_DIR=\"$PWD/tmp/profiling_repeatability\"

for i in 1 2 3; do
  RUN_DIR=$(pixi run -m \"$GSIM_REPO_ROOT\" vidur-cli svr init-run \
    model=llama2_7b hardware=a100 backend=sarathi workload=default vidur=default)
  pixi run -m \"$GSIM_REPO_ROOT\" vidur-cli svr profile --run-dir \"$RUN_DIR\"
done
```

Then compare the three:

- `<run_dir>/profile/data/profiling/cpu_overhead/.../cpu_overheads.csv`
