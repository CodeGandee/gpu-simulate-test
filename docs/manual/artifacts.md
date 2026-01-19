# Artifacts and schemas

This repo emits two families of artifacts, depending on which workflow you run:

- `001-compare-vidur-real-timing`: workload spec + real/sim metrics + a comparison report
- `002-reproduce-vidur-paper-fidelity`: canonical trace + sim/real paper-metric CSVs + a report

## A) Compare Vidur vs real timing (`001-compare-vidur-real-timing`)

### Workload spec (`tmp/workloads/<workload_id>/`)

- `prompts.jsonl`: prompt id + prompt text
- `trace_lengths.csv`: one row per request with token counts
  - required: `request_id`, `prompt_id`, `num_prefill_tokens`, `num_decode_tokens`
- `trace_intervals.csv`: one row per request with arrival schedule
  - required: `request_id`, `arrival_time_ns`, `inter_arrival_ns`
- `workload_meta.json`: resolved config + provenance snapshot

### Real run (`tmp/real_runs/<run_id>/`)

#### `request_metrics.csv`

One row per request.

Required columns (see `specs/001-compare-vidur-real-timing/contracts/request_metrics.md`):

- `request_id`
- `arrival_time_ns`
- `first_token_time_ns`
- `ttft_ns`
- `completion_time_ns`
- `num_prefill_tokens`
- `num_decode_tokens`
- `num_decode_tokens_actual`
- `status`

Notes:

- All timestamps are integer nanoseconds relative to run start (monotonic).
- If multiple requests share the same `arrival_time_ns` and the runner is sequential, `ttft_ns` will include queueing behind earlier requests.

#### `token_metrics.csv`

Long format: one row per decoded token.

Required columns (see `specs/001-compare-vidur-real-timing/contracts/token_metrics.md`):

- `request_id`
- `token_index`
- `token_time_ns`
- `token_latency_ns` (delta from previous token for that request)

### Vidur sim (`tmp/vidur_runs/<run_id>/`)

The standardized outputs match the real run schema:

- `request_metrics.csv`
- `token_metrics.csv`
- `run_meta.json`

Important: Vidur’s simulator produces request-level metrics and the wrapper derives a token timeline from them. For details, see `src/gpu_simulate_test/vidur_ext/sim_runner.py`.

MLP validation note:

- `run_meta.json` includes an `mlp_validation` summary for the consumed `mlp.csv` (mode, nan_policy, missing/zero-heavy stats).
- If the effective policy is `drop` (e.g., `vidur.validation.mlp.nan_policy=drop`), `run_meta.json` also includes `mlp_nan_drop` with per-op dropped-row counts used during sklearn training.

### Comparison report (`tmp/comparisons/<comparison_id>/`)

- `summary.md`: quick read (p50/p90/p99 tables)
- `tables/ttft_percentiles.csv`
- `tables/token_latency_percentiles.csv`
- `figs/*`: distribution plots

Token alignment note: comparisons truncate the sim series to match `num_decode_tokens_actual` from the real run (see `src/gpu_simulate_test/analysis/compare.py`).

## B) Paper fidelity reproduction (`002-reproduce-vidur-paper-fidelity`)

This workflow is trace-driven: a canonical `trace.csv` is the shared input to both Vidur simulation and Sarathi real replay.

### Trace (`tmp/paper_fidelity/traces/<scenario>/`)

- `trace.csv`: canonical trace input consumed end-to-end
  - required: `arrived_at` (seconds since start), `num_prefill_tokens`, `num_decode_tokens`
  - optional: `request_id`, `prompt_id`
- `trace_meta.json`: trace provenance (workload mode, seed, subset selection; may include trace source info and/or chosen dynamic QPS)

Trace subsetting is supported via config overrides:

- `trace_subset.kind=range trace_subset.begin=<b> trace_subset.end=<e>` for timed and untimed sources
- `trace_subset.kind=indices trace_subset.indices=[...]` only for untimed sources (those where arrivals are generated inside the workflow)

### Runs (`tmp/paper_fidelity/runs/<scenario>/`)

- `sim/`
  - `request_metrics.csv`: Vidur request-level paper metric columns (normalized metrics are preserved from Vidur raw outputs)
  - `vidur_raw/`: Vidur’s raw metric directory (for debugging)
  - `run_meta.json`
- `real/`
  - `request_metrics.csv`: Sarathi request-level paper metric columns (converted from Sarathi `sequence_metrics.csv`)
  - `sarathi/replica_0/sequence_metrics.csv` (raw Sarathi metrics)
  - `run_meta.json`
- `capacity/` (dynamic repro only)
  - `capacity.json`: discovered `capacity_qps` and `qps_85`, plus the overload criterion used
  - `qps_<x>/`: per-candidate Sarathi runs during search

Paper-fidelity request metric columns (required on both sim and real sides):

- `request_id`
- `request_scheduling_delay`
- `request_execution_plus_preemption_time_normalized`
- `request_e2e_time_normalized`
- `prefill_time_execution_plus_preemption_normalized`
- `decode_time_execution_plus_preemption_normalized`
- `request_num_decode_tokens`

### Report (`results/reports/<date>/paper_fidelity/<report_scenario>/`)

- `summary.md`: score tables + optional “gap diagnosis”
- `run_meta.json`: resolved config, provenance, and (when enabled) paper reference metadata
- `scores.json`: machine-readable score outputs (percentiles + percent error + verdict)
- `inputs/`: snapshots of inputs used to generate the report (portable; avoids `tmp/` reuse)
- `figs/`: SVG plots (ECDF + percentiles)
- `tables/`: derived tables (CSV)

Notes:

- The `<report_scenario>` component is derived from `scenario.name`:
  - static: `<scenario.name>`
  - dynamic: `<scenario.name>_dynamic_<scale>`
- Canonical sim/real CSVs live under `tmp/paper_fidelity/runs/<scenario.name>/{sim,real}/request_metrics.csv`. Reports also snapshot these CSVs under `inputs/` for reproducibility/portability.
- `tmp/paper_fidelity/runs/<scenario.name>/sim/run_meta.json` includes `mlp_validation` (and `mlp_nan_drop` when enabled) for the consumed profiling root.
