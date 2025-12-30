# `token_metrics.csv` Contract

Produced by:

- `/data1/huangzhe/code/gpu-simulate-test/tmp/real_runs/<run_id>/token_metrics.csv`
- `/data1/huangzhe/code/gpu-simulate-test/tmp/vidur_runs/<run_id>/token_metrics.csv`

All timestamps are integer nanoseconds relative to run start.

This is a long-format table: one row per *decode* token produced/represented.

## Required columns

- `request_id` (int): joins `request_metrics.csv`
- `token_index` (int): 0-based decode token index within the request
- `token_time_ns` (int): timestamp when the token is produced/considered complete
- `token_latency_ns` (int): for `token_index>0`, delta between consecutive `token_time_ns` values

## Optional columns

- `token_id` (int): token id if available (backend-dependent)

## Required invariants

- `token_index >= 0`
- `token_time_ns >= 0`
- For each `request_id`, `token_index` MUST be contiguous from `0..num_decode_tokens_actual-1` for real runs.
- For each `request_id`, `token_time_ns` MUST be non-decreasing with `token_index`.

## Notes on Vidur-derived values

If Vidur does not provide per-token timestamps, `vidur-sim` MAY generate token rows by expanding per-request average decode token latency and anchoring at workload `arrival_time_ns + ttft_ns` (see `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/research.md`).
