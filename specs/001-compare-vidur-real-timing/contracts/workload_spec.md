# Workload Spec Contract

Canonical workload spec directory:

- `/data1/huangzhe/code/gpu-simulate-test/tmp/workloads/<workload_id>/`

This directory is the single source of truth that drives both real benchmarking and Vidur simulation (via adapters).

## `prompts.jsonl`

**Type**: JSON Lines (one JSON object per line)

Required fields per line:

- `prompt_id` (int): 0-based
- `prompt` (string)

Optional fields:

- `meta` (object): arbitrary metadata (must be JSON-serializable)

## `trace_lengths.csv`

Required columns:

- `request_id` (int): 0-based
- `prompt_id` (int): joins `prompts.jsonl`
- `num_prefill_tokens` (int): >= 1
- `num_decode_tokens` (int): >= 1 (requested decode tokens)

Notes:

- `num_prefill_tokens` SHOULD be derived using the exact tokenizer/config specified in `workload_meta.json`.

## `trace_intervals.csv`

Required columns:

- `request_id` (int): 0-based (must match `trace_lengths.csv`)
- `arrival_time_ns` (int): >= 0, monotonic, relative to workload start
- `inter_arrival_ns` (int): >= 0, first row MUST be `0`

Invariants:

- `arrival_time_ns[i] == sum(inter_arrival_ns[: i + 1])`

## `workload_meta.json`

See JSON Schema:

- `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/contracts/workload_meta.schema.json`
