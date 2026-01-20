# Phase 0 Research: Vidur CLI Ray runtime configuration (env > config > defaults)

Date: 2026-01-20  
Spec: `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/spec.md`

This document resolves the Phase 0 research questions listed in `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/plan.md` and records key implementation decisions.

## Decision 1: Use Ray’s supported env vars as the “portable control plane”

Decision: Configure Ray’s default object-store sizing via environment variables and apply them before any stage that can import/start Ray.  
Rationale: Vidur and Sarathi start Ray internally (directly or indirectly). Ray itself reads these env vars when computing default object store memory, so setting env vars works without patching submodules or threading `ray.init(object_store_memory=...)` everywhere. This also keeps repo defaults opt-in (unset env vars → Ray defaults).  
Alternatives considered:
- Patch Vidur/Sarathi to pass `ray.init(object_store_memory=...)` everywhere (high maintenance; deep submodule edits).
- Call `ray.init(object_store_memory=...)` in `vidur-cli` before invoking submodules (works for max-bytes but doesn’t naturally support the “proportion” control and is more invasive at runtime).

Evidence (Ray implementation):
- `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES` and `RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION` are parsed in `/data1/huangzhe/code/gpu-simulate-test/.pixi/envs/default/lib/python3.13/site-packages/ray/_private/ray_constants.py`.
- Those defaults are used to derive `object_store_memory` when not explicitly set in `/data1/huangzhe/code/gpu-simulate-test/.pixi/envs/default/lib/python3.13/site-packages/ray/_private/resource_and_label_spec.py`.
- `/dev/shm` sizing guard and `RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE` behavior live in `/data1/huangzhe/code/gpu-simulate-test/.pixi/envs/default/lib/python3.13/site-packages/ray/_private/services.py`.

## Decision 2: Integration points — apply env defaults “as early as possible”

Decision: Apply + validate Ray env defaults in `vidur-cli` stage runners immediately after Hydra config composition and before any imports/calls that may touch Sarathi/Vidur/Ray.  
Rationale: Some helper utilities (e.g., CUDA pinning + Sarathi patching) can import Sarathi modules, which import Ray. Applying env first ensures settings are present for any subsequent Ray initialization, regardless of where `ray.init()` happens.  
Alternatives considered:
- Apply settings inside lower-level wrappers only (risk missing other Ray-starting call sites; harder to keep consistent across stages).
- Rely on Ray to ignore invalid env values (doesn’t satisfy “fail fast” requirement).

Concrete call sites to guard:
- Profiling stage: `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_cli/stages.py` `run_profile()` before calling `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_ext/profile_runner.py:run_vidur_profiling()` (subprocesses inherit env).
- Real replay stage (Sarathi backend): `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_cli/stages.py` `run_real()` before calling `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_cli/real_runner.py:run_token_length_replay()` (same process).

## Decision 3: Config shape — Hydra config group `ray/default.yaml` with nullable supported keys

Decision: Add a Hydra config group at `/data1/huangzhe/code/gpu-simulate-test/configs/compare_vidur_real/ray/default.yaml` with `ray.env` containing only the supported keys and nullable values.  
Rationale: Keeps repo defaults opt-in (`null` → no injection), provides a discoverable/overrideable config surface, and enables “unsupported key” detection by validating the dict keys.  
Alternatives considered:
- Flat keys like `ray_default_object_store_max_memory_bytes` (less explicit; loses direct mapping to env vars).
- Generic `Dict[str, Any]` with no validation (violates “fail fast on unknown keys”).

Supported keys (per spec):
- `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES` (int bytes)
- `RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION` (float proportion)
- `RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE` (bool)

## Decision 4: Validation + serialization rules (fail fast)

Decision: Validate both configuration values and pre-existing environment values for supported keys before starting any Ray-using stage; on invalid values, raise a user-facing error and do not start Ray.  
Rationale: The spec requires fail-fast, actionable errors. Ray itself may silently fall back to defaults for some env vars (e.g., `env_integer`), which is not sufficient.  
Alternatives considered:
- Let Ray’s parsing decide (can silently ignore invalid values; violates FR-007).

Rules:
- `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES`
  - Config: int, must be `>= 0`.
  - Env: string of digits (after stripping whitespace), must parse to int `>= 0`.
  - Serialization: set env var to `str(int_value)`.
- `RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION`
  - Config: float, must be `0 < value <= 1`.
  - Env: must parse as float, must satisfy `0 < value <= 1`.
  - Serialization: set env var to a non-scientific string if possible (e.g., `"0.3"`), otherwise `str(value)`; keep stable formatting in reports.
- `RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE`
  - Config: bool.
  - Env: accept `{ "1", "0", "true", "false" }` case-insensitive (reject other values).
  - Serialization: set env var to `"1"` when true, `"0"` when false (presence-only truthiness is not used to avoid ambiguous values).

Notes:
- Both sizing controls are allowed together and are not treated as a conflict (FR-012); Ray caps computed default memory using both max-bytes and proportion.
- Leaving a supported key unset in config must be a no-op (FR-004).

## Decision 5: Effective settings report — emit stable, user-readable output without changing stdout contract

Decision: Emit an “effective Ray settings report” for every Ray-using stage to **stderr** (so stdout remains the “primary output path” contract) and optionally persist a JSON artifact under the run directory for reproducibility.  
Rationale: Users need observability (FR-006) without breaking scripts that parse stdout. Persisting a JSON artifact helps compare host vs Docker runs.  
Alternatives considered:
- Print to stdout (risks breaking existing command-surface expectations).
- Only write a file (less discoverable during interactive debugging).

Proposed report content (stable across environments):
- For each supported key:
  - `effective` value (string or null for “Ray default/unset”)
  - `source`: `environment` | `configuration` | `default`
  - `env_present`: bool (whether the env var was set before applying config)

## Decision 6: “No-Ray compute profiling” implementation strategy

Decision: Implement no-Ray compute profiling by avoiding any imports of Vidur profiling `main.py` modules (they import `ray` at module import time) and instead run MLP profiling sequentially in-process using Vidur’s `MlpWrapper` (single-GPU only); skip attention profiling execution and always write the attention fallback CSV template.  
Rationale: The spec defines “no-Ray” as “no Ray at all” and requires fallback outputs for Ray-dependent profiling. Upstream Vidur exposes `--disable_ray` flags, but the current Vidur code still imports and uses Ray regardless, so relying on that flag does not satisfy the requirement.  
Alternatives considered:
- Use Vidur `--disable_ray` flags (insufficient because Ray is imported/used regardless).
- Keep using subprocess-based profilers but run Ray in local mode (still starts Ray; violates FR-014).
- Implement both MLP and attention sequentially (possible, but higher risk/complexity; initial scope prioritizes a safe minimal path).

Downstream compatibility requirements:
- Always produce `mlp.csv` and `attention.csv` under the profiling root in the same locations expected today by the sim-vs-real workflow (`/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_ext/profile_runner.py` already stages these paths).
- When skipping attention profiling, use the existing packaged-template fallback writer to produce a schema-compatible `attention.csv`.

