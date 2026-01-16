# Quickstart: MLP profiling with explicit method + validation

**Branch**: `005-vidur-mlp-cuda-driver`  
**Date**: 2026-01-16  
**Repo Root**: `/data1/huangzhe/code/gpu-simulate-test`

## 1) Prepare environment

1. Install/update the Pixi environment:

   - `cd /data1/huangzhe/code/gpu-simulate-test`
   - `pixi install`

2. Initialize submodules (required for Vidur):

   - `cd /data1/huangzhe/code/gpu-simulate-test`
   - `git submodule update --init --recursive`

3. Pin a healthy GPU for Torch/Ray workers:

   - `cd /data1/huangzhe/code/gpu-simulate-test`
   - create `/data1/huangzhe/code/gpu-simulate-test/.env` (not committed) with:
     - `export GSIM_CUDA_VISIBLE_DEVICES=0`

4. Sanity check CUDA in Pixi:

   - `cd /data1/huangzhe/code/gpu-simulate-test`
   - `pixi run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`

## 2) Run profiling with explicit MLP method selection

This feature requires the MLP profiling method to be selected explicitly via run configuration.

Examples (values shown are illustrative; adjust to your scenario):

- Paper-fidelity profiling run:
  - `cd /data1/huangzhe/code/gpu-simulate-test`
  - `pixi run paper-fidelity profile profiling.mlp.profile_method=cuda_event`

- Vidur profiling bundle export:
  - `cd /data1/huangzhe/code/gpu-simulate-test`
  - `pixi run vidur-profiling-bundle output.dir=/data1/huangzhe/code/gpu-simulate-test/tmp/vidur_bundle profiling.mlp.profile_method=cuda_event`
  - (Optional: exercise record-function attribution)
    - `pixi run vidur-profiling-bundle output.dir=/data1/huangzhe/code/gpu-simulate-test/tmp/vidur_bundle profiling.mlp.profile_method=record_function`

Note: some hosts require `GPU_SIMULATE_TEST_ENABLE_VIDUR_ATTENTION_COMPAT=1` for attention profiling
compatibility (applied via `src/sitecustomize.py`). The convenience script `pixi run vidur-profiling`
enables this by default.

## 3) Configure validation strictness and fallback

- Default strictness: strict (fail on missing values; fail on zero-heavy signals).
- Non-strict mode (warn on zero-heavy signals):
  - `... profiling.mlp.validation.mode=non_strict`
- Opt-in automatic fallback (retry with the alternate method if validation fails):
  - `... profiling.mlp.fallback.enabled=true`
  - `... profiling.mlp.fallback.method=cuda_event`

## 4) Inspect provenance

After a successful run, inspect the produced profiling meta record:

- `.../profiling_meta.json`

It should include:

- code revision identifier + local-change indicator
- environment snapshot
- resolved run configuration (`params`)
- profiling commands and output paths
- embedded MLP validation summary
