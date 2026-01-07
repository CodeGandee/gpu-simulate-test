# Behaviour: Warm-up requirements (profiling + real runs)

## Purpose

Ensure profiling data and real-inference measurements are not polluted by one-time initialization costs (e.g., model load, kernel autotune/compilation, allocator/JIT setup, first-request scheduler initialization).

## Requirements

### Profiling runs

- **Warm-up before measuring**: run warm-up iterations prior to collecting timing statistics.
- **Reset after warm-up**: clear timers/metrics/state used for statistics after warm-up and before the “measured” window starts.
- **Representative execution**: warm-up must exercise the same execution path as the measured runs for the target hardware + model + parallelism configuration (e.g., attention backend, TP degree).
- **Configurable**: warm-up iteration counts (or equivalent warm-up controls) must be configurable, with defaults that keep profiling stable and fast.

### Real inference runs (sim-vs-real comparisons)

- **Warm-up before replay**: execute a small warm-up request (or equivalent) before replaying the trace/workload.
- **Do not mix warm-up with the workload**: warm-up must not be included as a “trace request” and must not alter the workload’s arrival schedule.
- **Reset after warm-up**: reset metrics/timers after warm-up so the recorded request metrics correspond only to the workload requests.
- **Record provenance**: run metadata must indicate whether warm-up was enabled and which parameters were used (enough to interpret results and reproduce behaviour).
- **Default-on for comparisons**: for fidelity/scoring workflows, warm-up must be enabled by default; disabling warm-up is allowed only as an explicit override for debugging/experimentation.

## Rationale

- **First-run artifacts are large** relative to per-request latency at inference timescales, especially for smaller models where CPU overhead can dominate.
- **Fidelity evaluation is distribution-sensitive**: even a small number of slow “first” requests can skew tail percentiles (P95/P99), breaking paper-aligned comparisons.

## Observability

- The run artifacts should make it clear whether warm-up was used (via metadata), and the recorded metrics should reflect only the post-warm-up measurement window.

