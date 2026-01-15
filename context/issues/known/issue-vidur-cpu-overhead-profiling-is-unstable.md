# Issue: Vidur CPU overhead profiling is unstable (high run-to-run variance)

**Status**: Known
**Date**: 2026-01-15
**Last updated**: 2026-01-15
**Priority**: High (fidelity + reproducibility)

## Summary

Repeated runs of `vidur-cli svr profile` on the same host (same model/hardware) produce **materially different** CPU overhead profiling outputs (`cpu_overheads.csv`), even though:

- GPUs are pinned via `GSIM_CUDA_VISIBLE_DEVICES`
- Vidur’s profilers include warmup iterations
- Compute profiling outputs are stable (attention is bit-identical; MLP drifts slightly)

Because Vidur sim consumes these CPU overhead numbers, a `vidur-cli` sim-vs-real report (and any paper-fidelity run using a `vidur-cli` profiling root) can show noticeably different %errors depending on which profiling root is used.

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

For example, at `batch_size=16` (the default `backend.scheduler.max_num_seqs=16`):

| Metric (cpu_overheads.csv) | CV% across 3 runs |
|---|---:|
| `schedule_mean` | ~50.6% |
| `process_model_outputs_mean` | ~47.9% |
| `ray_comm_time_mean` | ~34.7% |

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

