# Hydra configs

This directory contains Hydra configuration groups used by `vidur-cli` and related workflows.

## `compare_vidur_real/`

Primary workflow config root used by `vidur-cli svr ...` stages.

### Ray runtime configuration (`ray` group)

Config group:

- `configs/compare_vidur_real/ray/default.yaml`

Supported keys (initial scope) under `ray.env.*`:

- `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES` (int | null, bytes, must be `>= 0`)
- `RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION` (float | null, must satisfy `0 < x <= 1`)
- `RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE` (bool | null)

Precedence is per setting:

1. Environment (`RAY_*` already set)
2. Configuration (`ray.env.*`)
3. Ray defaults (no injection)

Defaults are opt-in: the repo default config leaves these values `null`.

### No-Ray compute profiling (`vidur_profile` only)

`configs/compare_vidur_real/vidur_profile.yaml` supports:

- `profiling.compute.use_ray` (bool, default `true`)

Current status:

- `profiling.compute.use_ray=false` is **not supported yet** and will fail fast.
- Reason: in the tracked Vidur submodule, `--disable_ray` is currently a stub (the profiling scripts still
  call `ray.remote(...)` / `ray.get(...)`), and this repo intentionally does **not** hide missing attention
  profiling data by copying a pre-baked `attention.csv` template.
