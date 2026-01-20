# Manual smoke: `vidur-cli` no-Ray compute profiling (`profiling.compute.use_ray=false`)

This is a human-run smoke checklist for disabling Ray in Vidur compute profiling
(feature `specs/006-vidur-cli-ray-config/`).

## Prerequisites

- Initialize submodules: `git submodule update --init --recursive`
- Create the Pixi environment: `pixi install`
- Ensure CUDA is available: `pixi run python -c "import torch; print(torch.cuda.is_available())"`
- Optionally pin GPUs via `.env` (`GSIM_CUDA_VISIBLE_DEVICES=...`) per `context/instructions/prep-dev-env.md`

## Smoke: `svr profile` without starting Ray

From a scratch directory:

```bash
mkdir -p /tmp/vidur-cli-no-ray-profile-smoke
cd /tmp/vidur-cli-no-ray-profile-smoke

export GSIM_REPO_ROOT=/data1/huangzhe/code/gpu-simulate-test

RUN_DIR=$(
  pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr init-run \
    model=qwen3_0_6b hardware=a100 backend=transformers workload=default vidur=default
)
echo "RUN_DIR=$RUN_DIR"
```

Optional: capture Ray process state before:

```bash
pgrep -fa raylet || true
pgrep -fa plasma_store_server || true
```

Run compute profiling with Ray disabled:

```bash
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr profile --run-dir "$RUN_DIR" \
  profiling.compute.use_ray=false \
  profiling.mlp.profile_method=cuda_event
```

Expected:

- The command completes without starting Ray (no new `raylet`/`plasma_store_server` processes).
- Compute outputs exist under `$RUN_DIR/profile/`:
  - `data/profiling/compute/<hardware>/<model_id>/mlp.csv`
  - `data/profiling/compute/<hardware>/<model_id>/attention.csv` (fallback template)

Optional: verify Ray processes after:

```bash
pgrep -fa raylet || true
pgrep -fa plasma_store_server || true
```

## Unsupported cases (should fail fast)

No-Ray compute profiling currently rejects CPU overhead profiling:

```bash
set +e
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr profile --run-dir "$RUN_DIR" --include-cpu-overhead \
  profiling.compute.use_ray=false \
  profiling.mlp.profile_method=cuda_event
set -e
```
