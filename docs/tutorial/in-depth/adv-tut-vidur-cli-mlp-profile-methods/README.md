# Advanced tutorial: Vidur MLP `profile_method` sweep (sim-vs-real)

This tutorial helps answer: *“Why does sim-vs-real change so much when I change Vidur’s MLP profiling method?”*

You will run the exact same `vidur-cli` **static** sim-vs-real pipeline four times, varying only:

```yaml
profiling:
  mlp:
    profile_method: <one of: cuda_event | record_function | kineto | perf_counter>
```

and then compare the final report score tables.

All run artifacts go under `tmp/` (gitignored). This tutorial directory only tracks **small, essential** inputs and a
sanitized example of expected outputs.

## Concepts (what is `profiling.mlp.profile_method`?)

`profiling.mlp.profile_method` controls **how Vidur measures per-operation time** when building `mlp.csv` (the
microbenchmark dataset that feeds Vidur’s execution-time predictors).

Those predictors directly affect simulated request latency, so changing the profiling method can significantly shift
sim-vs-real accuracy.

Vidur supports four methods:

- `cuda_event`: times GPU work using CUDA events around operations (stable; usually the recommended default).
- `record_function`: uses `torch.profiler.record_function` + trace attribution (higher overhead).
  - Caveat: historically, this path could miss GPU time for kernels launched via the CUDA *driver* API; that was the
    motivating bug fixed by `005-vidur-mlp-cuda-driver`.
- `kineto`: uses `torch.profiler` (Kineto) and aggregates CUDA time from profiler events (high overhead; slow).
- `perf_counter`: wall-clock timing using `time.perf_counter()` with CUDA synchronizations (coarse; often overestimates
  due to sync/launch overhead).

## Step-by-step

### Step 0 — Prerequisites

From repo root:

```bash
pixi install
git submodule update --init --recursive
pixi run python -c "import torch; print(torch.cuda.is_available())"
bash models/llama2-7b-hf/bootstrap.sh
```

Expected result:
- CUDA check prints `True`.
- `models/llama2-7b-hf/source-data` exists (symlink to machine-local model assets).

### Step 1 — Run the sweep

From repo root:

```bash
docs/tutorial/in-depth/adv-tut-vidur-cli-mlp-profile-methods/run_sweep_static_profile_methods.sh
```

Expected result:
- The script prints `SWEEP_DIR=...` and finishes with:
  - `done: <SWEEP_DIR>`
  - `comparison: <SWEEP_DIR>/comparison.md`
- Under `<SWEEP_DIR>/`, you get one full `vidur-cli` run per method.

### Step 2 — Inspect per-method reports (sanity)

Open the copied summaries under `<SWEEP_DIR>/`:

- `<SWEEP_DIR>/cuda_event_summary.md`
- `<SWEEP_DIR>/record_function_summary.md`
- `<SWEEP_DIR>/kineto_summary.md`
- `<SWEEP_DIR>/perf_counter_summary.md`

Expected result:
- Each summary has a `## Profiling` section with an `mlp:` block, e.g.

```text
- mlp:
  - profile_method: `cuda_event`
  - validation: `mode=strict small_input_threshold=128 zero_heavy_limit=0.01`
  - fallback: `enabled=false method=cuda_event used=false`
```

This is how you confirm the report corresponds to the method you intended to test.

### Step 3 — Compare results across methods

Open:

- `<SWEEP_DIR>/comparison.md` (quick table)
- `<SWEEP_DIR>/comparison_scores.csv` (full score table)
- `<SWEEP_DIR>/comparison_runs.csv` (run dirs + settings)

Expected result:
- `comparison.md` contains:
  - a table mapping `method → run_dir`
  - a table of percent error for selected metrics (p50/p95)

### Step 4 — Interpret “sim is much larger than real”

If sim latencies are consistently **larger than real**, the simulator is likely using **over-estimated** microbenchmark
timings somewhere. In this tutorial you only vary MLP timing method, so:

- if `perf_counter` is worst: that’s often expected (sync-heavy wall-clock timing inflates costs)
- if `cuda_event` is worse than `record_function`/`kineto`: it suggests GPU-event timing for this setup is producing
  larger MLP costs than the trace-based methods (and you should choose the method that best matches your goal)

Use `comparison_scores.csv` to see whether the effect is mostly in:

- `prefill_time_*` vs `decode_time_*` (prefill-heavy vs decode-heavy sensitivity), and/or
- p50 vs p95 (tail sensitivity)

## Tracked inputs and expected outputs

This tutorial tracks only small files:

- `inputs/trace_import.csv`: static trace in `vidur-cli` import schema.
- `expected_outputs/`: sanitized example outputs to show the expected output *shape*.

Exact numbers vary across machines and code revisions; treat `expected_outputs/` as a structural reference.

## Maintainers: refresh `expected_outputs/`

Run:

```bash
docs/tutorial/in-depth/adv-tut-vidur-cli-mlp-profile-methods/run_sweep_static_profile_methods.sh --snapshot-expected
```

Expected result:
- A sanitized snapshot is written under `<SWEEP_DIR>/expected_snapshot/`.

To update the git-tracked expected outputs, copy the snapshot into:

- `docs/tutorial/in-depth/adv-tut-vidur-cli-mlp-profile-methods/expected_outputs/`
