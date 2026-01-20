# Quickstart: Ray runtime config in `vidur-cli` (Phase 1)

**Repo root (example)**: `/data1/huangzhe/code/gpu-simulate-test`  
**Feature**: `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/spec.md`  

This quickstart shows the intended user workflow for configuring a small set of Ray runtime settings via Hydra config (no manual `export RAY_*` required) and optionally disabling Ray for Vidur compute profiling.

## 0) Setup

From the repo root:

```bash
pixi install
```

## 1) Supported settings + precedence

Supported Ray settings (initial scope):

- `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES`
- `RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION`
- `RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE`

Precedence is per-setting:

1. Environment (`RAY_*` already set)
2. Workflow configuration (`cfg.ray.env.*`)
3. Ray defaults (no injection)

Repo defaults are opt-in: the default workflow config leaves all supported settings unset.

## 2) Configure Ray settings via Hydra overrides (recommended)

Example (apply settings without any `export RAY_*`):

```bash
pixi run vidur-cli svr profile --run-dir <run_dir> \
  ray.env.RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.10 \
  ray.env.RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=4000000000 \
  ray.env.RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=true
```

Notes:
- Unset/`null` values are treated as “leave to Ray defaults”.
- If the user already set an env var (e.g., `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES`), `vidur-cli` will not override it.

## 3) Docker-friendly example (avoid `/dev/shm` errors / spikes)

If Docker has a small `/dev/shm`, Ray may fail fast when the computed object store memory is large. Two knobs commonly used together are:

- Lower the default object store proportion (e.g., `0.10`)
- Allow falling back to slow storage (Ray uses a temp directory instead of `/dev/shm`) via `RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=true`

Example:

```bash
pixi run vidur-cli svr real --run-dir <run_dir> \
  backend=sarathi \
  ray.env.RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.10 \
  ray.env.RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=true
```

## 4) Respect user-set `RAY_*` env vars (power-user mode)

If you explicitly set an env var, it wins over config:

```bash
export RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.05
pixi run vidur-cli svr profile --run-dir <run_dir> \
  ray.env.RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.10
```

The effective settings report should show the env value as the source.

## 5) Disable Ray for compute profiling (no-Ray mode)

To disable Ray for Vidur compute profiling (initially single-GPU only):

```bash
pixi run vidur-cli svr profile --run-dir <run_dir> profiling.compute.use_ray=false
```

Expected behavior:
- The tool must not start Ray for compute profiling.
- Outputs required by downstream steps (`mlp.csv`, `attention.csv`) are still produced (attention may use a fallback template).
- Unsupported no-Ray configurations (e.g., multi-GPU) fail fast with an actionable error.

