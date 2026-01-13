# Quickstart: Paper-fidelity matrix for additional paper models

Spec: `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/spec.md`

## Prerequisites

- Repo root: `/data1/huangzhe/code/gpu-simulate-test`
- Submodules initialized:
  - `git submodule update --init --recursive`
- Pixi env installed:
  - `pixi install`
- CUDA works inside Pixi:
  - `pixi run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
- GPU pinning configured (required):
  - Set `GSIM_CUDA_VISIBLE_DEVICES` in `/data1/huangzhe/code/gpu-simulate-test/.env` (or export it in your shell).

## 1) Bootstrap model references (Sarathi needs local model assets)

Run per model (or `bash models/bootstrap.sh` to attempt all):

```bash
cd /data1/huangzhe/code/gpu-simulate-test

bash models/internlm-20b/bootstrap.sh
bash models/llama2-70b-hf/bootstrap.sh
bash models/qwen-72b/bootstrap.sh
```

Verify the `source-data` links exist:

```bash
ls -la /data1/huangzhe/code/gpu-simulate-test/models/internlm-20b/source-data
ls -la /data1/huangzhe/code/gpu-simulate-test/models/llama2-70b-hf/source-data
ls -la /data1/huangzhe/code/gpu-simulate-test/models/qwen-72b/source-data
```

## 2) Generate canonical traces (optional but recommended for validation)

This validates the scenario config can generate a valid trace for both workloads.

```bash
cd /data1/huangzhe/code/gpu-simulate-test

pixi run paper-fidelity trace --scenario internlm_20b_arxiv --workload static --scale small
pixi run paper-fidelity trace --scenario internlm_20b_arxiv --workload dynamic --scale small

pixi run paper-fidelity trace --scenario llama2_70b_arxiv --workload static --scale small
pixi run paper-fidelity trace --scenario llama2_70b_arxiv --workload dynamic --scale small

pixi run paper-fidelity trace --scenario qwen_72b_arxiv --workload static --scale small
pixi run paper-fidelity trace --scenario qwen_72b_arxiv --workload dynamic --scale small
```

Expected outputs (mutable, overwritten on reruns):
- Traces: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/traces/<scenario.name>/trace.csv`
- Trace meta: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/traces/<scenario.name>/trace_meta.json`

## 3) Run the required matrix (profile + static repro + dynamic repro at scale=small)

The matrix procedure must:
- run `paper-fidelity profile --include-cpu-overhead` for each model scenario,
- run `paper-fidelity repro` for `static` and `dynamic` at `--scale small` (50 requests),
- record failures with blocker categorization (including `insufficient GPUs`),
- write one per-matrix manifest summarizing all attempted runs (successes + failures).

Recommended interface for this feature (to be implemented):

```bash
cd /data1/huangzhe/code/gpu-simulate-test

# Example:
# - runs all three paper models
# - static + dynamic
# - small scale only
pixi run paper-fidelity matrix \
  --scale small \
  --scenarios internlm_20b_arxiv,llama2_70b_arxiv,qwen_72b_arxiv \
  --workloads static,dynamic \
  --include-cpu-overhead
```

## 4) Find outputs

### Profiling roots (per model, per run)

- `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/profiling_roots/<scenario.name>/<timestamp-dir>/`

### Reports (self-contained, stable)

Successful repro runs write report bundles under:

- `/data1/huangzhe/code/gpu-simulate-test/results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<scenario_name_or_tag>/`

Each report directory includes:
- `summary.md`
- `run_meta.json`
- `scores.json`
- `inputs/` snapshots (so reports are portable and not dependent on `tmp/`)

### Per-matrix manifest + failure records

The matrix procedure should write a dedicated output directory (example):

- `/data1/huangzhe/code/gpu-simulate-test/results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/paper_models_matrix_<run_id>/`
  - `manifest.json` (all attempted runs, including failures)
  - `failures/*.json` (structured failure records, one per failed action)

## 5) Manual fallback (single scenario)

If you want to run one model manually (useful for debugging), the minimal sequence is:

```bash
cd /data1/huangzhe/code/gpu-simulate-test

profiling_root="$(pixi run paper-fidelity profile --scenario llama2_70b_arxiv --include-cpu-overhead | tail -n 1)"

pixi run paper-fidelity repro --scenario llama2_70b_arxiv --workload static --scale small \
  "scenario.vidur.profiling_root=${profiling_root}"

pixi run paper-fidelity repro --scenario llama2_70b_arxiv --workload dynamic --scale small \
  "scenario.vidur.profiling_root=${profiling_root}"
```

