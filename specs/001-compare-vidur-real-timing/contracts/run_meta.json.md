# Contract: `run_meta.json`

**Scope**: written by real runs, Vidur runs, profiling runs, and comparison runs.

## Required top-level fields

- `schema_version` (string, e.g., `"v1"`)
- `run_id` (string)
- `run_type` (string): `"real" | "vidur" | "compare" | "vidur_profile"`
- `model` (string): model identifier (e.g., `"Qwen/Qwen3-0.6B"`)
- `started_at` (ISO-8601 string)
- `ended_at` (ISO-8601 string)
- `git_commit` (string, git SHA)
- `git_dirty` (bool, optional)
- `env` (object): environment snapshot (versions, device info, etc.)
- `params` (object): resolved run parameters (typically from Hydra config)
- `hydra` (object):
  - `config_path` (string)
  - `config_name` (string)

## Stage-specific fields

- For `run_type == "real"`:
  - `backend` (string): `"sarathi" | "transformers"`
  - `workload_dir` (string, absolute path)
- For `run_type == "vidur"`:
  - `workload_dir` (string, absolute path)
  - `profiling_root` (string, absolute path)
- For `run_type == "compare"`:
  - `real_run_dir` (string, absolute path)
  - `sim_run_dir` (string, absolute path)
- For `run_type == "vidur_profile"`:
  - `profiling_root` (string, absolute path)
