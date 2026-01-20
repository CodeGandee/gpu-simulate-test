# Manual smoke: `vidur-cli` Ray settings (`ray.env.*`)

This is a human-run smoke checklist for Ray runtime configuration in `vidur-cli`
(feature `specs/006-vidur-cli-ray-config/`).

## Prerequisites

- Initialize submodules: `git submodule update --init --recursive`
- Create the Pixi environment: `pixi install`
- For `svr profile` / Ray-backed `svr real` (GPU likely required):
  - Ensure CUDA is available: `pixi run python -c "import torch; print(torch.cuda.is_available())"`
  - Optionally pin GPUs via `.env` (`GSIM_CUDA_VISIBLE_DEVICES=...`) per `context/instructions/prep-dev-env.md`

## Smoke: config-only Ray settings (no `export RAY_*`)

From a scratch directory:

```bash
mkdir -p /tmp/vidur-cli-ray-settings-smoke
cd /tmp/vidur-cli-ray-settings-smoke

export GSIM_REPO_ROOT=/data1/huangzhe/code/gpu-simulate-test

# Ensure the supported env vars are not set (so config is the source).
unset RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES || true
unset RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION || true
unset RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE || true

RUN_DIR=$(
  pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr init-run \
    model=qwen3_0_6b hardware=a100 backend=sarathi workload=default vidur=default
)
echo "RUN_DIR=$RUN_DIR"
```

### `svr profile` (writes `profile/ray_settings.json`)

```bash
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr profile --run-dir "$RUN_DIR" \
  profiling.mlp.profile_method=cuda_event \
  ray.env.RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.10 \
  ray.env.RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=4000000000 \
  ray.env.RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=true
```

Expected:

- Stderr includes an effective settings report, e.g.:
  - `[ray-settings] stage=profile`
  - `[ray-settings] RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.1 source=configuration`
- File exists: `$RUN_DIR/profile/ray_settings.json`
- `run_state.json` records the absolute path:
  - `python -c "import json; p=json.load(open('$RUN_DIR/run_state.json')); print(p['artifacts']['profile']['ray_settings_json'])"`

### `svr real` (Ray backend only; writes `real/ray_settings.json`)

```bash
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr real --run-dir "$RUN_DIR" \
  backend=sarathi \
  ray.env.RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.10 \
  ray.env.RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=true
```

Expected:

- Stderr includes an effective settings report, e.g.:
  - `[ray-settings] stage=real`
  - `[ray-settings] RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1 source=configuration`
- File exists: `$RUN_DIR/real/ray_settings.json`
- `run_state.json` records the absolute path under `artifacts.real.ray_settings_json`

Notes:

- Precedence is per setting: env > config > default.
- Defaults are opt-in: `null` config values do not inject env vars and are reported as `effective=null`.
