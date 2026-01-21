# Known issue / pitfall: Vidur sim overestimates when attention profiling under-covers serving batch sizes

## Summary

In `vidur-cli` sim-vs-real runs, it is easy to accidentally run **attention profiling** on a much smaller batch-size
range than what the simulator/real replay actually uses.

When that happens, Vidur must **extrapolate** attention kernel time to larger batches, which can lead to a
**systematic overestimate** in simulated request latency.

Historical note: the tutorial originally kept `profiling.attention.max_batch_size` small for speed, which made this
pitfall easy to hit. The current tutorial defaults to matching the serving cap (16) and exposes an explicit env var
(`GSIM_VIDUR_ATTENTION_MAX_BATCH_SIZE`) so users can choose the speed/fidelity tradeoff deliberately.

## Where it shows up

Typical symptoms in `<run_dir>/report/summary.md`:

- `attention: ... min_batch_size=1 max_batch_size=<small>` (or generally `max_batch_size << batch_size_cap`)
- `request_execution_plus_preemption_time_normalized` and/or `request_e2e_time_normalized` have sim > real

## Root cause (what’s happening)

Vidur’s attention execution-time predictor is trained from the attention microbenchmark outputs.
If attention profiling only includes small decode batch sizes (e.g., only `batch_size=1`), but the actual run
regularly reaches larger decode batches (e.g., up to 16), the predictor has to extrapolate outside the observed
region. Depending on kernels/backends, this often biases toward **pessimistic** (slower) predictions.

This is especially common when:

- `profiling.attention.max_batch_size` is kept low for fast iteration, but
- the real/sim scheduler is configured with a much larger `max_num_seqs` / `batch_size_cap`

## How to confirm quickly

1) Check the report header for:

- `attention: ... max_batch_size=<N>`
- `batch_size: sim=<B> real=<B>` (or equivalent scheduler cap)

2) Inspect the attention profiling CSV under your profiling root and confirm the maximum `batch_size` profiled is
at least the maximum batch size your run can produce.

## Recommended parameter setup (rule of thumb)

For “apple-to-apple” fidelity runs:

1) **Cover the serving batch regime in attention profiling**

- Set: `profiling.attention.max_batch_size >= backend.scheduler.max_num_seqs`
- Keep: `profiling.attention.min_batch_size=1` (so the model also sees small-batch behavior)

2) **Keep KV-cache block/page size consistent**

- Ensure: `profiling.attention.block_size == backend.scheduler.block_size`

3) **If CPU overhead modeling is enabled, profile CPU overhead at relevant batch sizes too**

- Set: `profiling.cpu_overhead.max_batch_size >= backend.scheduler.max_num_seqs`
- Prefer running CPU overhead profiling on an otherwise idle host (it is noisier than compute profiling)

## Mitigations / workarounds

### Option A (recommended): re-profile attention with larger `max_batch_size`

Re-run the profiling stage with an override, e.g. for the tutorial defaults (`max_num_seqs=16`):

```bash
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr profile --run-dir "$RUN_DIR" \
  profiling.attention.max_batch_size=16
```

Then rerun `svr sim`, `svr real`, and `svr report`.

Tradeoff: attention profiling time increases with the batch-size grid.

### Option B: reduce the serving batch cap to match what you profiled

If you intentionally want to keep attention profiling at `max_batch_size=1`, set the scheduler batch caps to 1 so
the sim/real run never enters an out-of-distribution batch regime.

Tradeoff: you are no longer evaluating the original serving regime (latency/throughput will change).

### Option C: temporarily disable CPU overhead modeling (debugging only)

If you are diagnosing compute-only fidelity and want to eliminate CPU/runtime contributions, disable CPU overhead
profiling/modeling for the run (the tutorial demo script supports this via `GSIM_VIDUR_INCLUDE_CPU_OVERHEAD=false`).

Tradeoff: the sim is no longer trying to match end-to-end stack latency; interpret results accordingly.

## Notes

- For the tutorial demo script, set `GSIM_VIDUR_ATTENTION_MAX_BATCH_SIZE` if you want to trade fidelity for profiling
  speed (e.g., `1` for fast iteration; `16` to match the default serving cap).
- CPU overhead profiling/modeling can materially affect sim-vs-real and can be noisy; if you rely on it, consider
  running multiple CPU overhead profiles and using a robust aggregate.
