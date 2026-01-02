# Artifacts and schemas

This workflow standardizes outputs across the real run and the Vidur simulation to make comparisons straightforward.

## Workload spec (`tmp/workloads/<workload_id>/`)

- `prompts.jsonl`: prompt id + prompt text
- `trace_lengths.csv`: one row per request with token counts
  - required: `request_id`, `prompt_id`, `num_prefill_tokens`, `num_decode_tokens`
- `trace_intervals.csv`: one row per request with arrival schedule
  - required: `request_id`, `arrival_time_ns`, `inter_arrival_ns`
- `workload_meta.json`: resolved config + provenance snapshot

## Real run (`tmp/real_runs/<run_id>/`)

### `request_metrics.csv`

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

### `token_metrics.csv`

Long format: one row per decoded token.

Required columns (see `specs/001-compare-vidur-real-timing/contracts/token_metrics.md`):

- `request_id`
- `token_index`
- `token_time_ns`
- `token_latency_ns` (delta from previous token for that request)

## Vidur sim (`tmp/vidur_runs/<run_id>/`)

The standardized outputs match the real run schema:

- `request_metrics.csv`
- `token_metrics.csv`
- `run_meta.json`

Important: Vidur’s simulator produces request-level metrics and the wrapper derives a token timeline from them. For details, see `src/gpu_simulate_test/vidur_ext/sim_runner.py`.

## Comparison report (`tmp/comparisons/<comparison_id>/`)

- `summary.md`: quick read (p50/p90/p99 tables)
- `tables/ttft_percentiles.csv`
- `tables/token_latency_percentiles.csv`
- `figs/*`: distribution plots

Token alignment note: comparisons truncate the sim series to match `num_decode_tokens_actual` from the real run (see `src/gpu_simulate_test/analysis/compare.py`).

