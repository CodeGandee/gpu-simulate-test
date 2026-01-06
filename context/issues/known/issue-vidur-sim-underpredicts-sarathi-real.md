# Vidur sim underpredicts Sarathi real latency (common pattern)

## Summary

Vidur’s simulation can **systematically underpredict wall-clock latency** vs a real Sarathi-Serve run, especially when the simulation is driven primarily by **GPU-kernel-time-style profiling** and excludes host/runtime overhead.

This pattern can show up in:

- Paper-fidelity **sanity-check reproduction** (paper-provided profiling bundle): sim-vs-real `% error` on a different host/software stack is often **informational** only.
- Host-calibrated **gap reproduction**: `% error` is meaningful, but still sensitive to profiling success and runtime differences.

## Why this happens (common causes)

1. **CPU/runtime overhead is excluded**
   - If `skip_cpu_overhead_modeling=true`, the sim omits host-side overhead (scheduler, sampling, framework glue, synchronization, etc.).
   - In many regimes (including low batch sizes), this overhead can dominate real latency.

2. **Profiling bundle is not host-matched**
   - Paper-provided profiling was collected on the paper’s hardware/software stack.
   - Even with the same GPU SKU, differences in driver/CUDA/kernel, CPU, memory, kernel scheduling, and library versions can move real latency noticeably.

3. **Profiling fallback / mismatch**
   - If attention/compute profiling fails and falls back to a template, the sim may use unrealistic timings.
   - Mismatched scheduler knobs between sim and real (chunking, TP/PP, block size, etc.) can also shift latency.

## Mitigations

- For a meaningful sim-vs-real comparison on this machine:
  - Generate a host profiling root via `paper-fidelity profile` and override `scenario.vidur.profiling_root`.
- If you want the sim to account for host overhead:
  - Ensure the profiling root includes `cpu_overheads.csv` and set `skip_cpu_overhead_modeling=false`.
- If results look implausible:
  - Verify profiling subprocesses succeeded (no template fallbacks) and that sim/real configs match on key knobs.

## Evidence (example)

One concrete example of this underprediction pattern is documented here:

- `context/issues/known/issue-vidur-sim-underpredicts-sarathi-real-qwen3-0.6b.md`

That experiment is model-specific, but the underlying causes above generalize.

