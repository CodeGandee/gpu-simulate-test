# Data Model: Compare Vidur vs real Qwen3 A100 timing

**Feature**: `001-compare-vidur-real-timing`  
**Repo root**: `/data1/huangzhe/code/gpu-simulate-test`  
**Principle**: All persisted artifacts are filesystem-first and written under `/data1/huangzhe/code/gpu-simulate-test/tmp/` by default.

## Entities

### WorkloadSpec (directory)

**Path**: `/data1/huangzhe/code/gpu-simulate-test/tmp/workloads/<workload_id>/`

**Files**:
- `prompts.jsonl`: one JSON object per line, at minimum containing a stable `prompt_id` and prompt `text`.
- `trace_lengths.csv`: per-request token counts derived from the chosen tokenizer.
- `trace_intervals.csv`: per-request deterministic arrival schedule.
- `workload_meta.json`: provenance and config snapshot pointer.

### TraceLengthsRow (CSV row)

**File**: `trace_lengths.csv`  
**Primary key**: `request_id`

**Fields (minimum)**:
- `request_id`: integer, 0..N-1
- `prompt_id`: string (join key back to `prompts.jsonl`)
- `num_prefill_tokens`: integer (>= 0)
- `num_decode_tokens`: integer (>= 0; requested decode length)

### TraceIntervalsRow (CSV row)

**File**: `trace_intervals.csv`  
**Primary key**: `request_id`

**Fields (minimum)**:
- `request_id`: integer, 0..N-1
- `inter_arrival_ns`: integer nanoseconds (>= 0), relative to previous request
- `arrival_time_ns`: integer nanoseconds (>= 0), relative to workload start; must equal cumulative sum of `inter_arrival_ns`

### RunMetadata (JSON document)

**File**: `run_meta.json`  
**Used by**: real runs + Vidur runs + comparison runs

**Fields (minimum)**:
- `schema_version`: string (e.g., `"v1"`)
- `run_type`: string (`"real" | "vidur" | "compare" | "vidur_profile"`)
- `run_id`: string (directory name)
- `started_at`: ISO-8601 string
- `ended_at`: ISO-8601 string
- `git_commit`: string
- `git_dirty`: bool (optional)
- `model`: string
- `hardware`: object (optional; recommended)
- `hydra`: object: `{ "config_path": string, "config_name": string }`
- Stage-specific fields:
  - `workload_dir` (real/vidur)
  - `backend` (real)
  - `profiling_root` (vidur/vidur_profile)
  - `real_run_dir`, `sim_run_dir` (compare)

### RequestMetricsRow (CSV row)

**File**: `request_metrics.csv`  
**Primary key**: `request_id`

**Fields (minimum)**:
- `request_id`: integer
- `arrival_time_ns`: integer nanoseconds (relative to run start, monotonic)
- `first_token_time_ns`: integer nanoseconds (relative to run start, monotonic)
- `ttft_ns`: integer nanoseconds (>= 0)
- `num_prefill_tokens`: integer (>= 0)
- `num_decode_tokens`: integer (>= 0; requested decode length)
- `num_decode_tokens_actual`: integer (>= 0; may be < requested due to early stop)

**Optional fields (recommended)**:
- `prompt_id`: string

### TokenMetricsRow (CSV row, long format)

**File**: `token_metrics.csv`  
**Primary key**: (`request_id`, `token_idx`)

**Fields (minimum)**:
- `request_id`: integer
- `token_index`: integer (0..`num_decode_tokens_actual-1`)
- `token_time_ns`: integer nanoseconds (relative to run start, monotonic; timestamps at token emission)
 - `token_latency_ns`: integer nanoseconds (for `token_index > 0`, delta between consecutive `token_time_ns`)

**Derived fields (not required to persist)**:
- `token_latency_ns`: `token_time_ns[t] - token_time_ns[t-1]` for `t > 0` (if not already persisted)

### ComparisonReport (directory)

**Path**: `/data1/huangzhe/code/gpu-simulate-test/tmp/comparisons/<comparison_id>/`

**Files (minimum)**:
- `summary.md`: includes P50/P90/P99 for TTFT + per-token latency, and notes on alignment/early-stop handling
- `tables/*.csv`: percentile tables and/or CDF samples for programmatic review
- `figs/*.(png|pdf|html)`: plots for TTFT and per-token latency (real vs sim)
- `run_meta.json`: comparison provenance (points to both input run dirs)

## Invariants

- All timestamp columns use **integer nanoseconds** relative to run start (monotonic clock).
- `arrival_time_ns` in `trace_intervals.csv` must equal `cumsum(inter_arrival_ns)`.
- Comparisons must align per-token series using `num_decode_tokens_actual` (truncate simulator token series to actual real token count per request).
