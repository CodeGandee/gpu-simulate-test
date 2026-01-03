# Qwen3-0.6B: Vidur sim vs Sarathi real timing (A100)

## What this is

- Goal: run Vidur (simulation) and Sarathi-Serve (real inference) on the same workload so timings are meaningfully comparable.
- Key trick: the workload uses **fixed 2s inter-arrival** so `real-bench` (which runs requests sequentially) does not introduce queueing into TTFT.

## Run directories

- workload: `/data1/huangzhe/code/gpu-simulate-test/tmp/compare_experiments/20260102-134805Z-qwen3-0.6b-vidur-vs-sarathi-spaced-arrivals-2s/workload`
- real (sarathi): `/data1/huangzhe/code/gpu-simulate-test/tmp/compare_experiments/20260102-134805Z-qwen3-0.6b-vidur-vs-sarathi-spaced-arrivals-2s/real_run`
- sim (vidur): `/data1/huangzhe/code/gpu-simulate-test/tmp/compare_experiments/20260102-134805Z-qwen3-0.6b-vidur-vs-sarathi-spaced-arrivals-2s/vidur_run`
- compare: `/data1/huangzhe/code/gpu-simulate-test/tmp/compare_experiments/20260102-134805Z-qwen3-0.6b-vidur-vs-sarathi-spaced-arrivals-2s/compare` (`compare/summary.md` has percentile tables + plots)

## GPU vs CPU (what actually ran where)

- `real_run/` is **actual GPU inference** (device: `NVIDIA A100-SXM4-80GB`), measuring wall-clock after each `LLMEngine.step()`.
- `vidur_run/` is **CPU-side simulation** using an A100 profiling bundle + learned per-op timing models; it does **not** run real end-to-end GPU inference.

## Workload snapshot

`trace_intervals.csv` (arrivals):

| request_id | inter_arrival_ns | arrival_time_ns |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 2000000000 | 2000000000 |
| 2 | 2000000000 | 4000000000 |
| 3 | 2000000000 | 6000000000 |
| 4 | 2000000000 | 8000000000 |
| 5 | 2000000000 | 10000000000 |
| 6 | 2000000000 | 12000000000 |
| 7 | 2000000000 | 14000000000 |

`trace_lengths.csv` (token counts):

| request_id | prompt_id | num_prefill_tokens | num_decode_tokens |
| --- | --- | --- | --- |
| 0 | p0 | 9 | 64 |
| 1 | p1 | 9 | 64 |
| 2 | p2 | 14 | 64 |
| 3 | p3 | 13 | 64 |
| 4 | p4 | 13 | 64 |
| 5 | p5 | 11 | 64 |
| 6 | p6 | 7 | 64 |
| 7 | p7 | 10 | 64 |

No-overlap check (real run):

- `completion_time_ns[i] <= arrival_time_ns[i+1]` for all i: `True`

Submodule diff check (Vidur):

- `git -C extern/tracked/vidur status --porcelain` empty: `True`

## Key results (from `compare/tables/*.csv`)

TTFT percentiles:

| metric | real_ns | sim_ns | real_ms | sim_ms |
| --- | --- | --- | --- | --- |
| p50 | 17283962.500 | 1,967,610 | 17.284 | 1.968 |
| p90 | 20037415.300 | 1,967,610 | 20.037 | 1.968 |
| p99 | 22755979.330 | 1,967,610 | 22.756 | 1.968 |

Decode token latency percentiles:

| metric | real_ns | sim_ns | real_ms | sim_ms |
| --- | --- | --- | --- | --- |
| p50 | 13,611,908 | 1,824,346 | 13.612 | 1.824 |
| p90 | 14,125,543 | 1,828,523 | 14.126 | 1.829 |
| p99 | 17128720.200 | 1,828,524 | 17.129 | 1.829 |

## How to interpret the run CSVs

### `real_run/request_metrics.csv`

| request_id | arrival_time_ns | first_token_time_ns | ttft_ns | completion_time_ns | num_prefill_tokens | num_decode_tokens | num_decode_tokens_actual | status | backend |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 23058042 | 23058042 | 880144268 | 9 | 64 | 64 | ok | sarathi |
| 1 | 2000000000 | 2016598738 | 16598738 | 2885433132 | 9 | 64 | 64 | ok | sarathi |
| 2 | 4000000000 | 4016649246 | 16649246 | 4896383447 | 14 | 64 | 64 | ok | sarathi |
| 3 | 6000000000 | 6017364572 | 17364572 | 6890314297 | 13 | 64 | 64 | ok | sarathi |
| 4 | 8000000000 | 8017935641 | 17935641 | 8885079909 | 13 | 64 | 64 | ok | sarathi |
| 5 | 10000000000 | 10018742861 | 18742861 | 10891757119 | 11 | 64 | 64 | ok | sarathi |
| 6 | 12000000000 | 12016567178 | 16567178 | 12883875524 | 7 | 64 | 64 | ok | sarathi |
| 7 | 14000000000 | 14017203353 | 17203353 | 14892709310 | 10 | 64 | 64 | ok | sarathi |

### `vidur_run/request_metrics.csv`

| request_id | arrival_time_ns | first_token_time_ns | ttft_ns | completion_time_ns | num_prefill_tokens | num_decode_tokens | num_decode_tokens_actual | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 1967610 | 1967610 | 116835579 | 9 | 64 | 64 | ok |
| 1 | 2000000000 | 2001967610 | 1967610 | 2116835579 | 9 | 64 | 64 | ok |
| 2 | 4000000000 | 4001967610 | 1967610 | 4117164570 | 14 | 64 | 64 | ok |
| 3 | 6000000000 | 6001967610 | 1967610 | 6117098771 | 13 | 64 | 64 | ok |
| 4 | 8000000000 | 8001967610 | 1967610 | 8117098771 | 13 | 64 | 64 | ok |
| 5 | 10000000000 | 10001967610 | 1967610 | 10116967175 | 11 | 64 | 64 | ok |
| 6 | 12000000000 | 12001929755 | 1929755 | 12116666127 | 7 | 64 | 64 | ok |
| 7 | 14000000000 | 14001967610 | 1967610 | 14116901377 | 10 | 64 | 64 | ok |

### `real_run/token_metrics.csv` (request 0, first 8 tokens)

| request_id | token_index | token_time_ns | token_latency_ns | token_id |
| --- | --- | --- | --- | --- |
| 0 | 0 | 23058042 | 0 | 576 |
| 0 | 1 | 37930765 | 14872723 | 1714 |
| 0 | 2 | 51791541 | 13860776 | 304 |
| 0 | 3 | 65427637 | 13636096 | 279 |
| 0 | 4 | 79197622 | 13769985 | 2779 |
| 0 | 5 | 92709840 | 13512218 | 25 |
| 0 | 6 | 106287426 | 13577586 | 330 |
| 0 | 7 | 119934904 | 13647478 | 408 |

### `vidur_run/token_metrics.csv` (request 0, first 8 tokens)

| request_id | token_index | token_time_ns | token_latency_ns |
| --- | --- | --- | --- |
| 0 | 0 | 1967610 | 0 |
| 0 | 1 | 3790911 | 1823301 |
| 0 | 2 | 5614212 | 1823301 |
| 0 | 3 | 7437513 | 1823301 |
| 0 | 4 | 9260814 | 1823301 |
| 0 | 5 | 11084115 | 1823301 |
| 0 | 6 | 12907417 | 1823302 |
| 0 | 7 | 14730718 | 1823301 |

Note: Vidur only provides request-level timing in its raw metrics; this repo’s wrapper synthesizes per-token times by linear interpolation between `first_token_time_ns` and `completion_time_ns` (see `src/gpu_simulate_test/vidur_ext/sim_runner.py`).

## Reproduce

- Run: `bash /data1/huangzhe/code/gpu-simulate-test/tmp/compare_experiments/20260102-134805Z-qwen3-0.6b-vidur-vs-sarathi-spaced-arrivals-2s/scripts/run.sh`
- Regenerate this summary: `python3 /data1/huangzhe/code/gpu-simulate-test/tmp/compare_experiments/20260102-134805Z-qwen3-0.6b-vidur-vs-sarathi-spaced-arrivals-2s/scripts/make_summary.py`
