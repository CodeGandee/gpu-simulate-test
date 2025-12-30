# Quickstart: Compare Vidur vs real Qwen3 A100 timing

**Feature**: `001-compare-vidur-real-timing`  
**Date**: 2025-12-30  
**Spec**: `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/spec.md`

## Prerequisites

- Repo: `/data1/huangzhe/code/gpu-simulate-test`
- Submodules initialized: `git submodule update --init --recursive`
- Pixi env created: `pixi install`
- A100 available for real timing runs (for `torch.cuda.is_available()`).
- Qwen3 tokenizer/config reference available under:
  - `/data1/huangzhe/code/gpu-simulate-test/models/qwen3-0.6b/source-data/`

## Smoke checks (today, pre-feature)

- Vidur “it runs” smoke (from Vidur submodule root; CWD-dependent):
  - See `/data1/huangzhe/code/gpu-simulate-test/context/summaries/howto-use-vidur.md`
- Sarathi “it runs” smoke:
  - See `/data1/huangzhe/code/gpu-simulate-test/context/summaries/howto-use-sarathi-serve.md`

## End-to-end baseline (after implementation)

All outputs are written under `/data1/huangzhe/code/gpu-simulate-test/tmp/`.

1) Generate a deterministic workload:

```bash
pixi run workload-spec \
  --model Qwen/Qwen3-0.6B \
  --prompts /data1/huangzhe/code/gpu-simulate-test/tmp/prompts.jsonl \
  --out /data1/huangzhe/code/gpu-simulate-test/tmp/workloads/<workload_id>
```

2) Run the real benchmark (Qwen3 ground-truth via `transformers` backend):

```bash
pixi run real-bench \
  --backend transformers \
  --workload /data1/huangzhe/code/gpu-simulate-test/tmp/workloads/<workload_id> \
  --out /data1/huangzhe/code/gpu-simulate-test/tmp/real_runs/<run_id>
```

3) Generate Vidur profiling inputs (one-time per model+device):

```bash
pixi run vidur-profile \
  --model Qwen/Qwen3-0.6B \
  --profiling-root /data1/huangzhe/code/gpu-simulate-test/tmp/vidur_profiling
```

4) Run Vidur simulation from repo root:

```bash
pixi run vidur-sim \
  --workload /data1/huangzhe/code/gpu-simulate-test/tmp/workloads/<workload_id> \
  --profiling-root /data1/huangzhe/code/gpu-simulate-test/tmp/vidur_profiling \
  --out /data1/huangzhe/code/gpu-simulate-test/tmp/vidur_runs/<run_id>
```

5) Compare real vs sim and produce a report:

```bash
pixi run compare-runs \
  --real /data1/huangzhe/code/gpu-simulate-test/tmp/real_runs/<run_id> \
  --sim /data1/huangzhe/code/gpu-simulate-test/tmp/vidur_runs/<run_id> \
  --out /data1/huangzhe/code/gpu-simulate-test/tmp/comparisons/<run_id>
```

## Output checklist

- Workload:
  - `prompts.jsonl`, `trace_lengths.csv`, `trace_intervals.csv`, `workload_meta.json`
- Real:
  - `request_metrics.csv`, `token_metrics.csv`, `run_meta.json`, `summary.md`
- Vidur:
  - `request_metrics.csv`, `token_metrics.csv`, `run_meta.json`, `summary.md`
- Comparison:
  - `summary.md` + plots + percentile tables
