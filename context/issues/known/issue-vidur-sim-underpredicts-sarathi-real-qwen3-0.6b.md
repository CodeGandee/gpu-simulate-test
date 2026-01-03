# Vidur sim underpredicts Sarathi real latency (Qwen3-0.6B, A100, batch=1)

## Summary

In `tmp/compare_experiments/20260102-134805Z-qwen3-0.6b-vidur-vs-sarathi-spaced-arrivals-2s/`, Vidur simulation predicts **much lower** TTFT and decode per-token latency than a real Sarathi-Serve run on the same workload and GPU.

This is a **queue-free** baseline (requests are spaced so they do not overlap), so the gap is not explained by queueing/batching artifacts.

## Evidence (what was observed)

From `tmp/compare_experiments/20260102-134805Z-qwen3-0.6b-vidur-vs-sarathi-spaced-arrivals-2s/summary.md`:

- TTFT percentiles:
  - real p50: ~17.3 ms vs sim p50: ~2.0 ms
  - real p99: ~22.8 ms vs sim p99: ~2.0 ms
- Decode token latency percentiles:
  - real p50: ~13.6 ms vs sim p50: ~1.8 ms
  - real p99: ~17.1 ms vs sim p99: ~1.8 ms

Workload snapshot:

- 8 requests, ~7–14 prefill tokens, 64 decode tokens each.
- Inter-arrival spacing is 2s in `workload/trace_intervals.csv`.
- Real run shows no overlap: `completion_time_ns[i] <= arrival_time_ns[i+1]` for all requests.

## Repro

- Run end-to-end: `bash tmp/compare_experiments/20260102-134805Z-qwen3-0.6b-vidur-vs-sarathi-spaced-arrivals-2s/scripts/run.sh`
- Regenerate summaries: `python3 tmp/compare_experiments/20260102-134805Z-qwen3-0.6b-vidur-vs-sarathi-spaced-arrivals-2s/scripts/make_summary.py`

## Impact

- Absolute latency predicted by Vidur simulation is not reliable for this regime (single-stream, batch=1, short prompts, short generations).
- This can mislead decisions that depend on real wall-clock latency (SLO sizing, per-request overhead attribution, “what-if” comparisons).

## Suspected reasons (most likely contributors)

1. **CPU/runtime overhead is excluded in the sim configuration**
   - The Vidur sim wrapper forces `skip_cpu_overhead_modeling=True` (`src/gpu_simulate_test/vidur_ext/sim_runner.py`), and the profiling root validation defaults to skipping CPU overhead (`src/gpu_simulate_test/vidur_ext/profiling_root.py`).
   - In batch=1 regimes, real latency is often dominated by host-side scheduling, sampler, input prep, output processing, synchronization points, and framework overhead that a “GPU-kernel-time” model won’t capture.

2. **Attention profiling likely fell back to a template, producing unrealistic attention timings**
   - The profiling workflow can fall back to a template attention.csv if the attention profiling subprocess fails (`src/gpu_simulate_test/vidur_ext/profile_runner.py`).
   - The Qwen3 attention.csv in this run looks like a template-derived file (e.g., `attention_backend` entries such as `AttentionBackend.FLASH_ATTENTION`, which does not match Sarathi’s `AttentionBackend` enum in this repo), suggesting attention latency may be severely under-modeled.

3. **Compute profiling may be optimistic vs real end-to-end inference**
   - Vidur’s compute profiling uses dummy weights and a simplified profiling model structure (e.g., a single block repeated many times) which can be more cache-friendly and can underrepresent bandwidth/weight-traffic effects present in real inference.
   - The learned predictor uses limited features and a fixed batching-overhead fraction, which may not reflect batch=1 behavior well.

4. **Backend/config mismatch between “real” and “sim” execution**
   - Real run uses Sarathi engine settings like `chunk_size=16, max_num_seqs=1` (`src/gpu_simulate_test/real_bench/backends/sarathi_backend.py`), while the Vidur simulation config uses a different scheduler configuration (e.g., chunk size defaults in Vidur’s Sarathi scheduler config).
   - Even if queue-free, differences in iteration semantics, chunking, and kernel paths can shift per-token wall time significantly.

## Mitigations / next steps

- Enable and validate CPU overhead modeling in the Vidur sim path (requires producing and staging `cpu_overheads.csv`, and setting `skip_cpu_overhead_modeling=False`) to see how much of the gap is explained by host/runtime overhead.
- Ensure attention profiling succeeds for Qwen3-0.6B (avoid template fallback) and that the profiled attention backend matches the real backend used by Sarathi.
- Align “real” and “sim” scheduler knobs where possible (chunk size, block size, TP/PP) and rerun to quantify sensitivity.
- If the gap remains for batch=1, consider a calibrated constant per-step/per-token overhead term for “wall-clock” predictions in this regime (documented as a limitation).

## References

- Experiment: `tmp/compare_experiments/20260102-134805Z-qwen3-0.6b-vidur-vs-sarathi-spaced-arrivals-2s/summary.md`
- Compare report: `tmp/compare_experiments/20260102-134805Z-qwen3-0.6b-vidur-vs-sarathi-spaced-arrivals-2s/compare/summary.md`
- Related design notes: `context/tasks/working/001-compare-vidur-real-timing/qa-001-compare-vidur-real-timing.md`
