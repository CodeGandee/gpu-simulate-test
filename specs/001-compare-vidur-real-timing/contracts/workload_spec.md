# Contract: WorkloadSpec artifacts

**Scope**: artifacts under `/data1/huangzhe/code/gpu-simulate-test/tmp/workloads/<workload_id>/`

## Directory layout

```text
tmp/workloads/<workload_id>/
├── prompts.jsonl
├── trace_lengths.csv
├── trace_intervals.csv
└── workload_meta.json
```

## `prompts.jsonl`

- Format: JSON Lines (one JSON object per line)
- Required fields:
  - `prompt_id` (string): stable identifier referenced by trace CSVs
  - `text` (string): prompt content
- Optional fields: allowed (e.g., tags, source, language)

## `trace_lengths.csv`

- CSV header required
- One row per request; `request_id` must be stable and deterministic under fixed inputs.

Required columns:
- `request_id` (int): 0..N-1
- `prompt_id` (string): join key to `prompts.jsonl`
- `num_prefill_tokens` (int): >= 0
- `num_decode_tokens` (int): >= 0 (requested decode length)

## `trace_intervals.csv`

- CSV header required
- One row per request; must align 1:1 with `trace_lengths.csv` by `request_id`.
- All time values are integer nanoseconds relative to workload start.

Required columns:
- `request_id` (int): 0..N-1
- `inter_arrival_ns` (int): >= 0; delta from previous request (for request 0, may equal `arrival_time_ns`)
- `arrival_time_ns` (int): >= 0; must equal cumulative sum of `inter_arrival_ns`

## `workload_meta.json`

Required fields:
- `git_commit` (string, git SHA)
- `schema_version` (string, e.g., `"v1"`)
- `workload_id` (string)
- `created_at` (ISO-8601 string)
- `model` (string): e.g., `"Qwen/Qwen3-0.6B"`
- `seed` (int)
- `tokenizer_ref` (string): model/tokenizer reference used to compute token lengths (e.g., absolute path under `models/`)
- `paths` (object):
  - `prompts` (string, absolute path)
  - `trace_lengths` (string, absolute path)
  - `trace_intervals` (string, absolute path)
- `hydra` (object): `{ "config_path": string, "config_name": string }`
