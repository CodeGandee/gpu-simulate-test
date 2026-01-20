# Data Model: Ray runtime config + effective settings report (Phase 1)

**Feature**: `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/spec.md`  
**Date**: 2026-01-20  

This document defines the data entities introduced by the Ray runtime configuration feature, including configuration shapes and on-disk artifacts.

## Conventions

- All JSON artifacts include `schema_version: "v1"` at the top level.
- All paths stored in JSON are absolute, normalized paths.
- All timestamps are ISO-8601 UTC strings (e.g., `2026-01-20T08:30:00Z`).

## Entity: Workflow Configuration (Hydra)

**Purpose**: Allow `vidur-cli` workflows to set a small, supported set of Ray runtime settings and an option to disable Ray for compute profiling.

**Source**: Hydra composition via `vidur-cli` config roots (see `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_cli/search_path.py`).

### `ray.env` (supported Ray settings)

**Location in composed config**: `cfg.ray.env`

**Fields** (all optional / nullable; unset means “leave to Ray defaults”):
- `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES` (int | null)
- `RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION` (float | null)
- `RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE` (bool | null)

**Validation rules** (fail fast before starting Ray):
- `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES`: integer `>= 0`
- `RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION`: float `0 < x <= 1`
- `RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE`: boolean
- Unknown keys under `ray.env` are rejected (fail fast) and must list the supported keys.

### `profiling.compute.use_ray` (no-Ray compute profiling)

**Location in composed config**: `cfg.profiling.compute.use_ray` (in the `vidur_profile` config).

**Type**: bool (default `true`)

**Meaning**:
- `true`: run compute profiling as today (Ray may be used by Vidur profilers).
- `false`: do not start Ray for compute profiling; produce downstream-compatible profiling outputs via “no-Ray” execution + fallbacks (initially scoped to single-GPU).

## Entity: Ray Runtime Setting (logical)

**Purpose**: Represent a supported runtime knob with a value and provenance.

**Fields**:
- `key` (str): one of the supported env var names
- `effective_value` (str | null): the value that will be in effect for the stage (`null` means “Ray default / unset”)
- `source` (str): one of `environment`, `configuration`, `default`

Notes:
- “effective” is defined per setting, not per run (env may override config for one key but not others).

## Entity: Effective Settings Report (`ray_settings.json`)

**Purpose**: Make Ray runtime behavior observable to users and reproducible across host/Docker runs (FR-006).

**Location** (proposed):
- Profiling stage: `<run_dir>/profile/ray_settings.json`
- Real replay stage: `<run_dir>/real/ray_settings.json` (only when the chosen backend uses Ray)

**Fields (v1)**:
- `schema_version` (str, required): `"v1"`
- `created_at` (str, required): UTC ISO timestamp
- `stage` (str, required): e.g., `"profile"` or `"real"`
- `settings` (array[object], required): list of Ray Runtime Setting objects:
  - `key` (str, required)
  - `effective_value` (str | null, required)
  - `source` (str, required): `environment` | `configuration` | `default`

**Stability requirement**:
- If a setting is not explicitly set by env or config, `effective_value` should remain `null` (do not attempt to compute host-specific derived defaults), so reports remain comparable across host and Docker for the same config.

