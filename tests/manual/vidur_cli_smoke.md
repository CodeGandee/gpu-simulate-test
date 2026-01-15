# Manual smoke: `vidur-cli` (sim-vs-real workflow)

This is a human-run smoke checklist for `vidur-cli` (feature `specs/004-vidur-cli/`).

## Prerequisites

- Initialize submodules: `git submodule update --init --recursive`
- Create the Pixi environment: `pixi install`
- If running GPU stages (`profile/sim/real`):
  - Ensure CUDA is available: `pixi run python -c "import torch; print(torch.cuda.is_available())"`
  - Optionally pin GPUs via `.env` (`GSIM_CUDA_VISIBLE_DEVICES=...`) per `context/instructions/prep-dev-env.md`

## Smoke (CPU-only / fast)

Run from a scratch directory `<PWD>`:

```bash
mkdir -p /tmp/vidur-cli-smoke
cd /tmp/vidur-cli-smoke

export GSIM_REPO_ROOT=/data1/huangzhe/code/gpu-simulate-test

pixi run -m "$GSIM_REPO_ROOT" vidur-cli resources show
pixi run -m "$GSIM_REPO_ROOT" vidur-cli configs list --group model

RUN_DIR=$(
  pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr init-run \
    model=qwen3_0_6b hardware=a100 backend=transformers workload=default vidur=default
)
echo "RUN_DIR=$RUN_DIR"

cat > lengths.csv <<'EOF'
num_prefill_tokens,num_decode_tokens
8,16
12,16
EOF
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr trace --run-dir "$RUN_DIR" --from-lengths ./lengths.csv

# `svr sim` should fail fast if profiling hasn't been run yet, and write failure.json.
set +e
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr sim --run-dir "$RUN_DIR"
test -f "$RUN_DIR/failure.json"
set -e
```

Expected files under `$RUN_DIR` after CPU-only smoke:

- `run_state.json`
- `resources.json`
- `trace/trace.csv`
- `trace/trace_meta.json`
- `trace/trace_lengths.csv`
- `trace/trace_intervals.csv`
- `failure.json` (from the intentional `svr sim` failure)

## Smoke (full / GPU required)

This requires a working CUDA stack and local model assets for the selected preset.

```bash
export GSIM_REPO_ROOT=/data1/huangzhe/code/gpu-simulate-test

pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr profile --run-dir "$RUN_DIR"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr sim     --run-dir "$RUN_DIR"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr real    --run-dir "$RUN_DIR"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr report  --run-dir "$RUN_DIR"
```

Expected files after full smoke:

- `profile/` (profiling root layout under `data/profiling/...`)
- `sim/request_metrics.csv`, `sim/token_metrics.csv`
- `real/request_metrics.csv`, `real/token_metrics.csv`
- `report/summary.md` (must include arrival kind + CPU overhead status)

