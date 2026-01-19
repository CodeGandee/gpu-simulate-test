# Issue: Vidur MLP compute profiling misses CUDA driver-launched kernels (record_function) → 0s in `mlp.csv`

**Status**: Resolved in `main` for new profiling roots (existing roots may still contain placeholder 0.0s)
**Date**: 2026-01-16
**Last updated**: 2026-01-19
**Priority**: High (fidelity risk; silent underprediction)

## Summary

Some host profiling roots contain **0.0 timings** for compute ops in `mlp.csv` (e.g., `time_stats.attn_post_proj.{min,max,mean,median}=0`).

These are **not real “0 ms” kernels**. They originate from **missing per-op timing samples** in Vidur’s `record_function`-based profiler; historically those NaNs were **filled to 0.0 during staging**, masking the issue and biasing downstream training/simulation.

Because Vidur sim trains per-op timing predictors from `mlp.csv`, these 0s can cause the simulator to **systematically underpredict** compute time (and therefore latency / queueing).

## Fix status (2026-01-16)

Implemented in `005-vidur-mlp-cuda-driver`:

1. **Attribution fix (record_function)**: use a correlation-based tracer that counts both `cuda_runtime` and `cuda_driver` launch paths to attribute correlated `kernel` time back to `vidur_*` user-annotation regions.
2. **No more silent NaN → 0 staging**: staging validates `mlp.csv` and fails fast (strict by default) instead of masking missing values.
3. **Opt-in automatic fallback**: on validation failure, users can retry MLP profiling with an alternate method (e.g., `cuda_event`).
4. **Consumption validation**: consumers validate `mlp.csv` when loading a profiling root (strict fail vs non-strict warn), controlled by `vidur.validation.mlp.*`.

Code pointers (current implementation):

- Tracer: `src/gpu_simulate_test/vidur_ext/record_function_tracer_v2.py`
  - Correlates `cat=cuda_runtime` and `cat=cuda_driver` launches to `cat=kernel` execution events.
- MLP profiler wrapper patch: `src/gpu_simulate_test/vidur_ext/vidur_profiling_mlp_main.py`
  - Uses the v2 tracer when `profiling.mlp.profile_method=record_function`.
- Staging validation + optional fallback: `src/gpu_simulate_test/vidur_ext/profile_runner.py`
- Consumer validation: `src/gpu_simulate_test/vidur_ext/profiling_root.py`

## Where this shows up

The problematic values live in the staged profiling root:

- `<profiling_root>/data/profiling/compute/<device>/<model_id>/mlp.csv`

This impacts both:

- `paper-fidelity` runs that use such a profiling root
- `vidur-cli svr sim` runs that point at such a profiling root

## Evidence (example profiling root from paper-fidelity)

Example staged file with many zeros:

- `results/raw/2026-01-12/paper_fidelity/profiling_roots/llama2_7b_arxiv/2026-01-12_12-46-26-118841/data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/mlp.csv`

Observed characteristics (captured from local inspection):

- Total rows: 261 (token grid for `max_tokens=4096`)
- `time_stats.attn_post_proj.*` are exactly `0.0` in 85 rows
- Other ops also have 0s (e.g., `time_stats.mlp_up_proj.*` 36 rows, `time_stats.attn_pre_proj.*` 34 rows, …)

### Extracted CSV snippet (shows “missing → 0.0”)

Extracted subset of columns from the staged profiling root CSV:

- Source: `results/raw/2026-01-12/paper_fidelity/profiling_roots/llama2_7b_arxiv/2026-01-12_12-46-26-118841/data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/mlp.csv`

```csv
num_tokens,time_stats.attn_pre_proj.median,time_stats.attn_post_proj.median,time_stats.mlp_up_proj.median,time_stats.mlp_down_proj.median
4096,1.724608,0.0021915,3.0434125000000005,1.449027
3872,1.660034,0.0,2.9437025,1.4438765
3072,1.2898139999999998,0.422829,2.2745755,1.1406785000000002
2048,0.8821540000000001,0.0,1.5236375,0.768618
512,0.0,0.099103,0.0,0.25731
256,0.0,0.0621119999999999,0.0,0.136847
128,0.080415,0.0,0.16139,0.0979194999999999
```

Same extracted subset from the *raw* profiler output (before staging) shows the corresponding fields were actually empty (NaN), not 0:

- Source (untracked tmp): `tmp/paper_fidelity/profiling_outputs/llama2_7b_arxiv/2026-01-12_12-46-26-118841/mlp/2026-01-12_12-46-32/meta-llama/Llama-2-7b-hf/mlp.csv`

```csv
num_tokens,time_stats.attn_pre_proj.median,time_stats.attn_post_proj.median,time_stats.mlp_up_proj.median,time_stats.mlp_down_proj.median
4096,1.7246080000000001,0.0021915,3.0434125000000005,1.449027
3872,1.660034,,2.9437024999999997,1.4438765
2048,0.8821540000000001,,1.5236375,0.768618
512,,0.09910300000000001,,0.25731
256,,0.062111999999999994,,0.136847
128,0.08041500000000001,,0.16139,0.09791949999999999
```

Quick check (no pandas required):

```bash
python3 - <<'PY'
import csv
from pathlib import Path

p = Path("results/raw/2026-01-12/paper_fidelity/profiling_roots/llama2_7b_arxiv/2026-01-12_12-46-26-118841/data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/mlp.csv")
with p.open(newline="") as f:
    r = csv.reader(f)
    header = next(r)
    idx = {k: i for i, k in enumerate(header)}
    total = 0
    zeros = 0
    for row in r:
        total += 1
        if float(row[idx["time_stats.attn_post_proj.median"]]) == 0.0:
            zeros += 1
print("rows=", total, "attn_post_proj.median==0 rows=", zeros)
PY
```

### Why these rows became 0 in the staged profiling root

In the **raw MLP profiler output** (before staging), these entries are typically **empty** (NaN), not 0.

For the same profiling run (`run_id=2026-01-12_12-46-26-118841`), the raw output path recorded in `profiling_meta.json` is:

- `tmp/paper_fidelity/profiling_outputs/llama2_7b_arxiv/2026-01-12_12-46-26-118841/mlp/.../meta-llama/Llama-2-7b-hf/mlp.csv`

Example symptom seen during inspection:

- `num_tokens=3872` had `time_stats.attn_post_proj.*` cells as empty strings (`''`).

Historically, our staging wrapper masked the issue by filling missing `time_stats.*` cells to `0.0`.
This avoided downstream training crashes on NaNs, but silently converted “missing measurement” into
“0 ms”.

With `005-vidur-mlp-cuda-driver`, this behavior is removed: staging validates `mlp.csv` and fails
fast (strict by default), with an opt-in automatic fallback rerun.

## Root cause

Vidur’s `record_function` tracer only attributes time from `cat == "cuda_runtime"` events:

- `extern/tracked/vidur/vidur/profiling/utils/record_function_tracer.py:71`

However, for some GEMM shapes/kernels, the forward pass can be launched via the CUDA **driver** API:

- `cat == "cuda_driver"` events like `cuLaunchKernel`
- paired with a `cat == "kernel"` event (same correlation id)

When the only “launch-ish” event under the user annotation is `cuda_driver`, Vidur’s tracer sees **no correlated cuda_runtime → kernel pair**, so the op gets **no samples**, producing NaNs in the raw `mlp.csv`.

## Impact on simulation

Vidur’s execution-time predictor trains per-op regressors from `mlp.csv`:

- `extern/tracked/vidur/vidur/execution_time_predictor/sklearn_execution_time_predictor.py:451`
  - targets include `time_stats.attn_post_proj.median`, `time_stats.mlp_up_proj.median`, etc.

If missing measurements are staged as 0.0:

1) The trained model can learn that some ops are ~0 ms at those token counts.
2) Simulation will underpredict batch compute time.
3) Underpredicted compute time can also reduce simulated queueing/scheduling delays, amplifying error in sim-vs-real reports.

## Mitigations (for existing profiling roots)

- Re-profile using an explicit MLP method override (e.g., `profiling.mlp.profile_method=cuda_event`).
- If a run fails validation, enable opt-in fallback: `profiling.mlp.fallback.enabled=true profiling.mlp.fallback.method=cuda_event`.
- If a consumer fails on load, the error message should include the affected columns and remediation actions.

## How to verify you are no longer affected

If you are generating a new profiling root:

- Ensure the run uses an explicit method selection (required now): `profiling.mlp.profile_method=...`.
- Ensure `mlp.csv` passed strict validation (the run would fail otherwise).
- Check the report/metadata includes the resolved MLP settings:
  - `mlp.profile_method`
  - `mlp.validation.*`
  - `mlp.fallback.*` (and whether fallback was used)

If you are consuming an existing profiling root:

- If the root was produced before this fix, consumption in strict mode may fail and point at missing/zero-heavy columns.
  Re-profile with `profiling.mlp.profile_method=cuda_event` (or enable fallback) to produce a new root.

## Fix implemented (code)

1) **Record-function attribution**: correlation-based tracer counts both `cuda_runtime` and `cuda_driver` launch paths and attributes correlated `kernel` time to `vidur_*` regions.
2) **Fail fast instead of masking**: staging validates `mlp.csv` (missing always fails; zero-heavy fails in strict, warns in non-strict).
3) **Provenance**: profiling meta records the selected method, fallback usage, and validation summary.

## Implementation plan

- `context/plans/plan-fix-vidur-mlp-profiling-cuda-driver-kernels.md`
