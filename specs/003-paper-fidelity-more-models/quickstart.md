# Quickstart: Paper-fidelity sweep for additional paper models

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

## 3) Run the sweep (profile + static repro + dynamic repro at scale=small)

Use the sweep script to run profiling + repro for each case. It also supports global TP/PP overrides.

Example:

```bash
cd /data1/huangzhe/code/gpu-simulate-test

bash scripts/paper_fidelity_sweep.sh \
  --scale small \
  --workloads static,dynamic \
  --tp 1 \
  --pp 1 \
  --run-id my_run_001
```

Notes:
- `--scenarios` is optional; if omitted, the script defaults to the paper-model set:
  `internlm_20b_arxiv,llama2_70b_arxiv,qwen_72b_arxiv` (Qwen3-0.6B is excluded from the default set).
- Profiling includes CPU overhead microbenchmarks by default (pass `--no-include-cpu-overhead` to disable).
- `--stop-on-failure` stops after the first failed action.

## 4) Find outputs

### Profiling roots (per model, per run)

- `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/profiling_roots/<scenario.name>/<timestamp-dir>/`

The sweep script overrides `scenario.name` to include the run id:

- `tmp/paper_fidelity/profiling_roots/<scenario_key>_sweep_<run_id>/<timestamp-dir>/`

### Reports (self-contained, stable)

Successful repro runs write report bundles under:

- `/data1/huangzhe/code/gpu-simulate-test/results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<scenario_name_or_tag>/`

Each report directory includes:
- `summary.md`
- `run_meta.json`
- `scores.json`
- `inputs/` snapshots (so reports are portable and not dependent on `tmp/`)

### Sweep log

The sweep script writes an append-only log:

- `/data1/huangzhe/code/gpu-simulate-test/results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/sweep_<run_id>/cases.jsonl`

### Failure records (schema + blocker categories)

Failure records are always written as JSON with schema version `v1`.

**Where to find them**

- Repro failure (single-scenario `paper-fidelity repro`):
  - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<scenario_name_or_tag>/failure_record.json`
- Profiling failure (single-scenario `paper-fidelity profile`):
  - `tmp/paper_fidelity/profiling_roots/<scenario.name>/<timestamp-dir>/failure_record.json`

**Key fields**

- `action`: `trace|profile|repro`
- `scenario_key`: the scenario key under `configs/paper_fidelity/scenario/`
- `scenario_name`: the artifact/report namespace (may include `_sweep_<run_id>`)
- `workload`: `static|dynamic` (nullable for profiling)
- `scale`: `small|medium|full` (nullable for profiling)
- `attempted_command`: exact argv list (when available)
- `error_message` / `traceback`: failure details
- `blocker_category` (one of):
  - `insufficient GPUs`
  - `OOM`
  - `missing model files`
  - `unsupported model`
  - `unknown`

**Common remediation**

- `insufficient GPUs`: adjust `GSIM_CUDA_VISIBLE_DEVICES`, or override `scenario.real.parallel.{tensor,pipeline}_parallel_size` to fit.
- `missing model files`: run the model bootstrap script and re-check `models/<model>/source-data`.
- `OOM`: reduce parallelism, batch sizes, or choose a smaller model.

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
