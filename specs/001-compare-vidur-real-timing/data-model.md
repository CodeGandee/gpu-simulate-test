# Data Model: Compare Vidur vs real Qwen3 A100 timing

**Feature**: `001-compare-vidur-real-timing`  
**Date**: 2025-12-30  
**Spec**: `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/spec.md`

This feature is file/artifact-driven. “Entities” map directly to on-disk artifacts under `tmp/`.

## Entity: WorkloadSpec (directory)

**Location**: `/data1/huangzhe/code/gpu-simulate-test/tmp/workloads/<workload_id>/`

Fields (stored in `workload_meta.json`):

- `workload_id` (string): Stable identifier for the workload directory.
- `model` (string): e.g. `Qwen/Qwen3-0.6B`.
- `seed` (int): RNG seed used for deterministic generation.
- `tokenizer_ref` (string): e.g. `/data1/huangzhe/code/gpu-simulate-test/models/qwen3-0.6b/source-data/`.
- `created_at` (string, ISO8601).
- `git_commit` (string): commit hash for provenance.

Related files:

- `prompts.jsonl`: prompt set (JSON objects).
- `trace_lengths.csv`: request-level token lengths.
- `trace_intervals.csv`: request arrival schedule.

Identity:

- `workload_id` is globally unique within `tmp/workloads/`.

## Entity: Prompt (row in `prompts.jsonl`)

Fields:

- `prompt_id` (int): 0-based.
- `prompt` (string): raw text.
- `meta` (object, optional): any extra fields (source, tags).

Relationship:

- `Prompt.prompt_id` joins to `TraceLengths.prompt_id`.

## Entity: TraceLengths (row in `trace_lengths.csv`)

Fields:

- `request_id` (int): 0-based.
- `prompt_id` (int): 0-based, references `prompts.jsonl`.
- `num_prefill_tokens` (int): tokenizer-derived prompt length (>=1).
- `num_decode_tokens` (int): requested decode tokens (>=1).

Relationship:

- `request_id` joins to `TraceIntervals.request_id`.
- `prompt_id` joins to `Prompt.prompt_id`.

## Entity: TraceIntervals (row in `trace_intervals.csv`)

Fields:

- `request_id` (int): 0-based.
- `arrival_time_ns` (int): nanoseconds since workload start.
- `inter_arrival_ns` (int): delta from previous request (first row `0`).

Invariants:

- `arrival_time_ns` is non-decreasing.
- `arrival_time_ns[i] == sum(inter_arrival_ns[: i + 1])` (consistency-by-construction).

## Entity: RunMetadata (file: `run_meta.json`)

**Location**:

- Real: `/data1/huangzhe/code/gpu-simulate-test/tmp/real_runs/<run_id>/run_meta.json`
- Vidur: `/data1/huangzhe/code/gpu-simulate-test/tmp/vidur_runs/<run_id>/run_meta.json`

Fields:

- `run_id` (string)
- `run_type` (string enum): `real` | `vidur`
- `backend` (string enum, real only): `sarathi` | `transformers`
- `model` (string)
- `workload_dir` (string, absolute path)
- `hardware` (object): e.g. `{"gpu_name": "...", "gpu_count": 1}`
- `started_at` / `ended_at` (string, ISO8601)
- `git_commit` (string)
- `env` (object): key runtime versions (python/torch/cuda/driver)
- `params` (object): run knobs (dtype, batch limits, scheduler knobs, etc.)

Identity:

- `run_id` is globally unique within `tmp/{real_runs,vidur_runs}/`.

## Entity: RequestMetrics (row in `request_metrics.csv`)

**Location**:

- Real: `/data1/huangzhe/code/gpu-simulate-test/tmp/real_runs/<run_id>/request_metrics.csv`
- Vidur: `/data1/huangzhe/code/gpu-simulate-test/tmp/vidur_runs/<run_id>/request_metrics.csv`

Fields:

- `request_id` (int)
- `arrival_time_ns` (int)
- `first_token_time_ns` (int)
- `ttft_ns` (int): `first_token_time_ns - arrival_time_ns`
- `completion_time_ns` (int, optional): timestamp of last token (or request end)
- `num_prefill_tokens` (int)
- `num_decode_tokens` (int): requested
- `num_decode_tokens_actual` (int): produced (real) or requested (vidur proxy)
- `status` (string enum): `ok` | `error`
- `error_type` / `error_msg` (string, optional)

Relationship:

- `request_id` joins to `TokenMetrics.request_id`.

## Entity: TokenMetrics (row in `token_metrics.csv`)

**Location**:

- Real: `/data1/huangzhe/code/gpu-simulate-test/tmp/real_runs/<run_id>/token_metrics.csv`
- Vidur: `/data1/huangzhe/code/gpu-simulate-test/tmp/vidur_runs/<run_id>/token_metrics.csv`

Fields:

- `request_id` (int)
- `token_index` (int): 0-based decode token index.
- `token_time_ns` (int): monotonic timestamp when token is produced/considered complete.
- `token_latency_ns` (int): delta from previous token completion time (for `token_index==0`, this may equal `ttft_ns` or be left blank depending on the chosen convention).
- `token_id` (int, optional): token id if available (real runs; backend-dependent).

## Entity: ComparisonReport (directory)

**Location**: `/data1/huangzhe/code/gpu-simulate-test/tmp/comparisons/<run_id>/`

Artifacts:

- `summary.md`: narrative + percentile tables.
- Plot images (CDF/percentile curves) for TTFT and per-token latency.
- Optional CSV tables with computed percentiles for reproducibility.

Relationship:

- References one real run directory and one vidur run directory.
