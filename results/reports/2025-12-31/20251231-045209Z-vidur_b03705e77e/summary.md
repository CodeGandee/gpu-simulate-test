# Vidur run summary: `vidur_b03705e77e`

## Overview

- **Model**: `Qwen/Qwen3-0.6B`
- **Hardware preset**: `a100` (`cuda:0`)
- **Workload**: `tmp/workloads/workload_b3572b3a53`
- **Profiling root**: `tmp/vidur_profiling/a100/qwen3_0_6b`
- **Run output**: `tmp/vidur_runs/vidur_b03705e77e`

## Environment (from `run_meta.json`)

- **Started at**: `2025-12-31T04:52:09.826603+00:00`
- **Ended at**: `2025-12-31T04:52:16.387633+00:00`
- **Git commit**: `0bf647dc9e6bea06afd7fbd2390807cf2c6597b9` (dirty: `true`)

## Outputs

- Standardized metrics:
  - `tmp/vidur_runs/vidur_b03705e77e/request_metrics.csv`
  - `tmp/vidur_runs/vidur_b03705e77e/token_metrics.csv`
  - `tmp/vidur_runs/vidur_b03705e77e/run_meta.json`
- Vidur raw metrics directory:
  - `tmp/vidur_runs/vidur_b03705e77e/vidur_raw/2025-12-31_04-52-14-496148/`

## Key results (this run)

All timing fields below are **integer nanoseconds** relative to run start.

- Important: these metrics are produced by **Vidur’s CPU-side simulation** (using profiled/per-op models), not by running real end-to-end Qwen3 inference kernels on the GPU. If you want actual GPU timings, run `pixi run real-bench ...` and compare against this run.

- **Requests**: `4`
- **Total prefill tokens**: `53`
- **Total decode tokens**: `256`
- **TTFT** (`ttft_ns`): `2,204,013 ns` (`2.204 ms`) (min/median/max all the same)
- **E2E** (`completion_time_ns - arrival_time_ns`): `121,967,428 ns` (`121.967 ms`) (min/median/max all the same)
- **Per-token decode latency** (`token_latency_ns`, token_index>0):
  - mean: `1,901,006.6 ns` (`1.901 ms`)
  - median: `1,901,007 ns` (`1.901 ms`)
  - min/max: `1,901,006 ns` / `1,901,007 ns`
- **Approx decode throughput**: `~2098.9 tokens/s` over `0.121967 s` (max completion time window)

## How to interpret the CSV outputs

### `request_metrics.csv` (one row per request)

Columns (standardized by `src/gpu_simulate_test/vidur_ext/sim_runner.py`):

- `request_id`: 0-based request index (matches `trace_lengths.csv` / `trace_intervals.csv` in the workload dir).
- `arrival_time_ns`: when the request arrives (from workload `trace_intervals.csv`).
- `first_token_time_ns`: when the first decode token is produced (`arrival_time_ns + ttft_ns`).
- `ttft_ns`: time-to-first-token (`first_token_time_ns - arrival_time_ns`).
- `completion_time_ns`: when the request finishes (`arrival_time_ns + request_e2e_time` from Vidur, converted to ns).
- `num_prefill_tokens`, `num_decode_tokens`: planned tokens from the workload spec.
- `num_decode_tokens_actual`: actual decode tokens (may differ if early-stop is modeled; here it matches `num_decode_tokens`).
- `status`: `ok` for successful completion.

Example (request 0):

| request_id | arrival_time_ns | first_token_time_ns | ttft_ns | completion_time_ns | num_prefill_tokens | num_decode_tokens | num_decode_tokens_actual | status |
|-----------:|----------------:|--------------------:|--------:|-------------------:|------------------:|-----------------:|-------------------------:|:-------|
| 0 | 0 | 2,204,013 | 2,204,013 | 121,967,428 | 12 | 64 | 64 | ok |

Interpretation:
- `ttft_ns=2,204,013` means the first generated token is at `~2.204 ms` after arrival.
- End-to-end latency is `completion_time_ns - arrival_time_ns = 121,967,428 ns` (`~121.967 ms`).

Tip: to map `request_id -> prompt_id -> text`, join with the workload outputs:
- `tmp/workloads/workload_b3572b3a53/trace_lengths.csv` (has `prompt_id`)
- `tmp/workloads/workload_b3572b3a53/prompts.jsonl` (has the prompt text)

### `token_metrics.csv` (one row per decoded token)

Columns:

- `request_id`: which request this token belongs to.
- `token_index`: 0-based decode token index (`0` is the first generated token).
- `token_time_ns`: absolute token completion time (ns since run start).
- `token_latency_ns`: delta from previous token time (`0` for `token_index=0`).

Important: for Vidur runs, `token_metrics.csv` is **synthesized** by interpolating between
`first_token_time_ns` and `completion_time_ns`, so per-token latencies are expected to look nearly constant.

Example (request 0, first 6 decode tokens):

| request_id | token_index | token_time_ns | token_latency_ns |
|-----------:|------------:|--------------:|-----------------:|
| 0 | 0 | 2,204,013 | 0 |
| 0 | 1 | 4,105,020 | 1,901,007 |
| 0 | 2 | 6,006,026 | 1,901,006 |
| 0 | 3 | 7,907,033 | 1,901,007 |
| 0 | 4 | 9,808,039 | 1,901,006 |
| 0 | 5 | 11,709,046 | 1,901,007 |

## Per-request snapshot

| request_id | num_prefill_tokens | num_decode_tokens_actual | ttft_ms | e2e_ms |
|-----------:|-------------------:|-------------------------:|--------:|-------:|
| 0 | 12 | 64 | 2.204 | 121.967 |
| 1 | 10 | 64 | 2.204 | 121.967 |
| 2 | 15 | 64 | 2.204 | 121.967 |
| 3 | 16 | 64 | 2.204 | 121.967 |
