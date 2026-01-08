# Performance metrics definitions (contracts)

This doc defines the semantics of **request-level latency metrics** used in this repo (and aligned with the Vidur paper’s fidelity evaluation).

## Scope

- These are **per-request** metrics derived from timestamps/durations collected from either a real serving run or a simulation run.
- Aggregations like `p50`, `p95`, `p99`, etc. are computed over a population of requests for a given scenario (model + workload + config + load point).

## Core fields (per request)

The following fields are treated as the “source of truth” for computing derived metrics:

- `arrived_at`: time the request enters the system.
- `scheduled_at`: time the request is first admitted for execution (starts service).
- `prefill_completed_at`: time prefill finishes (the request transitions from prefill to decode).
- `completed_at`: time the request finishes all tokens (prefill + decode) and exits.
- `execution_time`: cumulative time spent actually executing request work (sum of per-stage/per-iteration execution durations), excluding waiting.
- `preempted_time`: cumulative time the request is *paused* due to preemption (time between a stage/iteration completion and the next time it is scheduled again).
- `scheduling_delay`: queueing delay before the request is first scheduled (`scheduled_at - arrived_at`).
- `num_prefill_tokens`: input (prompt) token count.
- `num_decode_tokens`: output token count (length of generated sequence); must be `> 0` for normalized metrics.

## Derived metrics (per request)

### `request_e2e_time_normalized` (dynamic fidelity metric)

**Meaning:** *Normalized end-to-end latency* (“per output token” latency), including queueing delay and all execution/preemption effects.

- Definition:
  - `request_e2e_time = completed_at - arrived_at`
  - `request_e2e_time_normalized = request_e2e_time / num_decode_tokens`
- Units: seconds per output token (often reported as ms/token).
- Paper mapping:
  - The paper compares **percentage error** for **normalized end-to-end latency** on **dynamic workloads** (near capacity) and plots `p50`/`p95` of this metric.
  - See `extern/tracked/vidur/paper/tex/5-eval.tex:37` and figure inputs like `extern/tracked/vidur/paper/tex/figures-tex/fig-fidelity-dynamic-trace.tex:5`.

### `request_execution_plus_preemption_time_normalized` (static fidelity metric)

**Meaning:** *Normalized “execution” latency* (“per output token” service time), excluding initial queueing delay but including time lost to preemption.

This is designed for **static-workload fidelity** where queueing/scheduling delay would dominate and obscure execution-time prediction quality.

- Definition:
  - `request_execution_plus_preemption_time = execution_time + preempted_time`
  - `request_execution_plus_preemption_time_normalized = request_execution_plus_preemption_time / num_decode_tokens`
- Units: seconds per output token (often ms/token).
- Interpretation:
  - Excludes initial queueing delay (`scheduling_delay`) and focuses on “how long the system spent servicing the request once it started”, while still counting preemption-induced pauses as part of service.
- Paper mapping:
  - The paper’s static fidelity figure uses “normalized execution latency” with filenames containing `request_execution_plus_preemption_time_normalized` (p50/p95).
  - See `extern/tracked/vidur/paper/tex/5-eval.tex:37` and `extern/tracked/vidur/paper/tex/figures-tex/fig-fidelity-static-trace.tex:5`.

## Percentiles and naming

When the paper (and this repo) reports `p50`/`p95`, it means:

- `p50`: the median request value for that metric (50th percentile).
- `p95`: the 95th-percentile request value (**tail latency**); 95% of requests are at or below this value, and the slowest (most time-consuming) ~5% of requests are above it.
  - Equivalently: `p95` characterizes the **slow tail**, not the fastest/bottom 5%.

Percentiles are computed on the per-request metric values *within the scenario being evaluated* (same trace/workload, model, scheduler/config, and load point).

## Invariants / validation rules

- `completed_at >= arrived_at`.
- `scheduled_at >= arrived_at`.
- `prefill_completed_at` is within `[scheduled_at, completed_at]` when present.
- `num_decode_tokens > 0` is required for:
  - `request_e2e_time_normalized`
  - `request_execution_plus_preemption_time_normalized`
- If any of the source-of-truth fields are missing, the derived metric is considered undefined for that request and must be excluded from percentile aggregation (or the run must fail fast, depending on the pipeline stage).

## Reference implementations (non-normative)

These are code locations that currently implement the above semantics:

- Vidur request timing definitions: `extern/tracked/vidur/vidur/entities/request.py:113`
- Vidur metric emission: `extern/tracked/vidur/vidur/metrics/metrics_store.py:510`
