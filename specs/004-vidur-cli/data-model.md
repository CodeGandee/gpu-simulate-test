# Data Model: Vidur CLI artifacts (Phase 1)

**Feature**: `specs/004-vidur-cli/spec.md`  
**Date**: 2026-01-14  

This document defines the on-disk entities (files) created and consumed by `vidur-cli`. It is the source of truth for validating artifacts and writing tests.

## Conventions

- All JSON artifacts include `schema_version: "v1"` at the top level.
- All paths stored in JSON are absolute, normalized paths (after expansion).
- All timestamps are ISO-8601 UTC strings (e.g., `2026-01-14T08:30:00Z`).

## Entity: Resource Map (`resources.json`)

**Purpose**: Record resolved resources and where each value came from (env / config TOML / repo fallback / default).

**Location**: `<run_dir>/resources.json`

**Fields (v1)**:
- `schema_version` (str, required): `"v1"`
- `resolved_at` (str, required): UTC ISO timestamp
- `repo_root` (object, required):
  - `value` (str, required): absolute path
  - `source` (str, required): one of `env`, `config_toml`, `repo_fallback`, `pwd_default`
  - `details` (object, optional): e.g., env var name or TOML key
- `models_root` (object, required): same shape as `repo_root`
- `datasets_root` (object, required): same shape as `repo_root`
- `workspace_root` (object, required):
  - `value` (str, required): absolute path to workspace root
  - `source` (str, required): one of `env`, `config_toml`, `pwd_default`
- `config_toml` (object, required):
  - `path` (str, required): absolute path to the resolved config TOML
  - `source` (str, required): one of `flag`, `env`, `default`
- `hydra_config_roots` (array[str], required): ordered list of config directories actually used
- `env_snapshot` (object, optional): selected environment variables relevant to resolution

## Entity: Run State (`run_state.json`)

**Purpose**: The canonical state machine for a single sim-vs-real run. Each stage updates this file.

**Location**: `<run_dir>/run_state.json`

**Identity rules**:
- `run_tag` is unique within a workspace root.
- `run_dir` is derived from workspace root + `run_tag` unless explicitly provided.

**Fields (v1)**:
- `schema_version` (str, required): `"v1"`
- `created_at` (str, required): UTC ISO timestamp
- `updated_at` (str, required): UTC ISO timestamp (last modification)
- `run_tag` (str, required): filesystem-safe tag (`preset+timestamp` by default)
- `run_dir` (str, required): absolute path to the run directory root
- `presets` (object, required):
  - `model` (str, required)
  - `hardware` (str, required)
  - `backend` (str, required)
  - `workload` (str, required)
  - `vidur` (str, required)
- `overrides` (array[str], required): trailing Hydra-style `key=value` overrides applied for this run
- `artifacts` (object, required):
  - `trace` (object, optional):
    - `trace_csv` (str, required): absolute path
    - `trace_meta_json` (str, required): absolute path
    - `status` (str, required): one of `ok`, `failed`
    - `ended_at` (str, required)
  - `profile` (object, optional):
    - `profiling_root` (str, required): absolute path
    - `include_cpu_overhead` (bool, required)
    - `status` (str, required): one of `ok`, `failed`
    - `ended_at` (str, required)
  - `sim` (object, optional):
    - `sim_run_dir` (str, required): absolute path
    - `status` (str, required): one of `ok`, `failed`
    - `ended_at` (str, required)
  - `real` (object, optional):
    - `real_run_dir` (str, required): absolute path
    - `backend` (str, required)
    - `status` (str, required): one of `ok`, `failed`
    - `ended_at` (str, required)
  - `report` (object, optional):
    - `report_dir` (str, required): absolute path
    - `summary_md` (str, required): absolute path
    - `status` (str, required): one of `ok`, `failed`
    - `ended_at` (str, required)

**Lifecycle/state transitions**:
- `init-run` creates `run_state.json` with `created_at=updated_at` and empty `artifacts`.
- Each stage appends or overwrites `artifacts.<stage>` and refreshes `updated_at`.
- On failure, the stage writes `failure.json` (see below) and sets `artifacts.<stage>.status="failed"` when possible.

## Entity: Stage Failure (`failure.json`)

**Purpose**: Preserve actionable error details without deleting partial outputs.

**Location**: `<run_dir>/failure.json` (overwritten by the most recent failure)

**Fields (v1)**:
- `schema_version` (str, required): `"v1"`
- `failed_at` (str, required): UTC ISO timestamp
- `stage` (str, required): one of `init-run`, `trace`, `profile`, `sim`, `real`, `report`, `resources`, `configs`
- `error_type` (str, required): exception class name (or `"RuntimeError"`)
- `message` (str, required): human-readable message
- `context` (object, optional): key-value diagnostics (missing prerequisite paths, attempted sources, etc.)

## Entity: Canonical Trace (`trace/trace.csv`, `trace/trace_meta.json`)

**Purpose**: The canonical input dataset for sim-vs-real stages (shared by profiling/sim/real/report steps).

**Location**:
- `<run_dir>/trace/trace.csv`
- `<run_dir>/trace/trace_meta.json`

**Trace CSV schema**:
- Required columns:
  - `request_id` (int, unique)
  - `arrival_time_ns` (int, >= 0, non-decreasing)
  - `num_prefill_tokens` (int, >= 1)
  - `num_decode_tokens` (int, >= 1)
- Optional columns:
  - Any extra columns are permitted and preserved (ignored by runners, stored for provenance).

**Trace meta JSON fields (v1)**:
- `schema_version` (str, required): `"v1"`
- `created_at` (str, required): UTC ISO timestamp
- `trace_csv` (str, required): absolute path
- `source` (object, required):
  - `kind` (str, required): one of `import`, `lengths_csv`
  - `path` (str, optional): absolute path of the imported file
- `arrival_schedule` (object, required):
  - `kind` (str, required): `fixed_interval` or `poisson`
  - `seed` (int, required)
  - `inter_arrival_ns` (int, optional)
  - `poisson_rate_per_s` (float, optional)

## Directory layout (run workspace)

Within `<run_dir>`:

```text
<run_dir>/
├── run_state.json
├── resources.json
├── resolved_config.yaml        # optional snapshot for provenance
├── failure.json                # written only on failure
├── trace/
│   ├── trace.csv
│   └── trace_meta.json
├── profile/
│   └── ... stage outputs ...
├── sim/
│   └── ... stage outputs ...
├── real/
│   └── ... stage outputs ...
└── report/
    ├── summary.md
    └── ... figs/tables ...
```
