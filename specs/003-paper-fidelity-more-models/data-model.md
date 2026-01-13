# Data Model: Paper-fidelity more models

Spec: `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/spec.md`  
Scope: paper-fidelity workflow for InternLM-20B, LLaMA2-70B, Qwen-72B (excluding Qwen3-0.6B)

## Entities

### Scenario (Hydra config)

**Source of truth:** YAML under `/data1/huangzhe/code/gpu-simulate-test/configs/paper_fidelity/scenario/`

**Key fields**
- `scenario_key` (string): filename stem (e.g., `llama2_70b_arxiv`)
- `name` (string): artifact namespace; becomes folder name under `tmp/paper_fidelity/...` and default report name
- `model.model_id` (string): Vidur/Sarathi model identifier (e.g., `meta-llama/Llama-2-70b-hf`)
- `model.model_ref` (path): local model asset path (e.g., `/data1/huangzhe/code/gpu-simulate-test/models/llama2-70b-hf/source-data`)
- `trace_source`:
  - `kind` (enum): `vidur_processed_lengths_csv` | `trace_csv` | `legacy_workload_dir`
  - `path` (path): input trace source
  - `max_tokens` (int): cap/filter for lengths
  - `seed` (int): deterministic trace generation
  - `num_requests` (int|null): optional cap
- `vidur`:
  - `profiling_root` (path): profiling root used by Vidur (overridden to host-matched root at runtime)
  - `device` / `network_device` (string): modeling choices
  - `tensor_parallel_size` (int)
  - `num_pipeline_stages` (int)
  - `skip_cpu_overhead_modeling` (bool)
  - `cpu_overhead.validation` (enum): `strict` | `warn` | `off`
  - `scheduler` (object): parity-critical knobs (`chunk_size`, `batch_size_cap`, `block_size`, ...)
- `real`:
  - `backend` (string): `sarathi`
  - `scheduler.chunk_size` (int)
  - `scheduler.max_num_seqs` (int)
  - `parallel.tensor_parallel_size` (int)
  - `parallel.pipeline_parallel_size` (int)
- `capacity_search` (object): enabled + bounded search params (dynamic runs)
- `scoring` (object): percentiles + thresholds
- `paper_reference` (object): optional paper row lookup for display

**Validation rules (minimum)**
- `name` is non-empty
- `model.model_ref` exists and is loadable by the real backend
- `trace_source.path` exists
- `tensor_parallel_size >= 1`, `num_pipeline_stages >= 1`
- For matrix execution, if `required_gpus = tensor_parallel_size * pipeline_parallel_size` and available GPUs < `required_gpus`, fail fast with blocker category `insufficient GPUs`

---

### Trace Artifact

**Produced by:** `paper-fidelity trace` and/or `paper-fidelity repro`

**Files**
- `trace.csv` (path): canonical trace (static: all arrivals at 0; dynamic: timed arrivals)
- `trace_meta.json` (path): schema `v1` with `scenario_name`, `workload_mode`, `scale`, subset info, seed, and artifact pointers

**Validation rules**
- CSV has required columns: `arrived_at`, `num_prefill_tokens`, `num_decode_tokens`, `request_id`
- `trace_meta.json` exists and matches the trace subset / seed used

---

### Profiling Root

**Produced by:** `paper-fidelity profile`

**Fields**
- `profiling_root` (path): directory containing `data/profiling/...`
- `profiling_meta.json` (path): provenance + profiling outputs summary
- `include_cpu_overhead` (bool): required `true` for this feature’s acceptance matrix
- `expected_parallelism` (int): derived from scenario TP/PP

**Validation rules**
- `profiling_root/data/profiling/` exists
- If CPU overhead modeling is intended to be enabled, CPU overhead CSV exists under the expected path for the scenario’s network device + model id

---

### Paper-fidelity Run (Test Run)

One execution of a workflow action for a specific scenario/workload/scale.

**Fields**
- `run_id` (string): stable identifier or timestamp-based id for a matrix run entry
- `action` (enum): `profile` | `repro`
- `workload` (enum|null): `static` | `dynamic` (null for profiling)
- `scale` (enum|null): `small` | `medium` | `full` (null for profiling)
- `scenario_key` (string)
- `scenario_name` (string): output namespace (often overridden to include matrix run id)
- `status` (enum): `success` | `failure`
- `started_at` / `ended_at` (ISO8601 UTC strings)
- `artifacts`:
  - on success: `report_dir` (path) and/or `profiling_root` (path)
  - on failure: `failure_record_json` (path)

**State transitions**
- `planned` → `running` → (`success` | `failure`)

---

### Report Bundle

**Produced by:** scoring/report step during `paper-fidelity repro`

**Directory**
- `report_dir`: `/data1/huangzhe/code/gpu-simulate-test/results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<scenario_name>/`

**Core files**
- `summary.md`
- `run_meta.json`
- `scores.json`
- `inputs/` snapshots:
  - `sim_request_metrics.csv`
  - `real_request_metrics.csv`
  - `trace.csv` (if available)
  - `trace_meta.json` (if available)
  - `capacity.json` (dynamic only, if available)

**Validation rules**
- Core files exist
- Inputs are present so the report is portable and not dependent on `tmp/` paths

---

### Failure Record

**Produced by:** matrix runner (and optionally by individual commands) when a required run fails.

**Fields (minimum)**
- `schema_version` (string): e.g. `v1`
- `generated_at` (ISO8601 UTC string)
- `run_id` (string)
- `action` (enum): `profile` | `trace` | `repro` | `matrix`
- `scenario_key` / `scenario_name` (string)
- `workload` / `scale` (nullable)
- `attempted_command` (list[string] or string)
- `hydra_overrides` (list[string])
- `error_message` (string)
- `traceback` (string|null)
- `blocker_category` (enum): `insufficient GPUs` | `OOM` | `missing model files` | `unsupported model` | `unknown`

**Validation rules**
- `blocker_category` is one of the allowed categories
- Failure record path is included in the per-matrix manifest

---

### Matrix Manifest

Single JSON file summarizing all attempted runs in the matrix (successes + failures).

**Fields**
- `schema_version` (string): e.g. `v1`
- `generated_at` (ISO8601 UTC string)
- `run_id` (string): matrix run id
- `scenarios` (list[string])
- `workloads` (list[enum])
- `scale` (enum)
- `runs` (list[object]): each run includes status, artifact pointers, and (for failures) failure details or a pointer to the failure record
- provenance: git commit + dirty flag + env snapshot

**Validation rules**
- Includes all required model/workload combinations attempted
- For successful runs, `report_dir` exists and contains core artifacts
- For failed runs, failure record exists and includes blocker category + error message

