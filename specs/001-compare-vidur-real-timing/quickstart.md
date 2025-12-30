# Quickstart: Compare Vidur vs real Qwen3 A100 timing (Hydra configs)

**Feature**: `001-compare-vidur-real-timing`  
**Repo root**: `/data1/huangzhe/code/gpu-simulate-test`

## Setup

```bash
cd /data1/huangzhe/code/gpu-simulate-test
git submodule update --init --recursive
pixi install
```

## End-to-end baseline (workload → real → sim → report)

> These commands assume Pixi tasks exist for each stage and each stage is a Hydra app with presets under `/data1/huangzhe/code/gpu-simulate-test/configs/compare_vidur_real/`.

### 1) Generate workload spec (deterministic)

```bash
pixi run workload-spec \
  workload.prompts=/data1/huangzhe/code/gpu-simulate-test/tmp/prompts/example.prompts.jsonl \
  workload.seed=123 \
  model=qwen3_0_6b
```

**Outputs**: `/data1/huangzhe/code/gpu-simulate-test/tmp/workloads/<workload_id>/`

### 2) Run real benchmark on A100 (choose backend)

```bash
pixi run real-bench \
  backend=transformers \
  workload.workload_dir=/data1/huangzhe/code/gpu-simulate-test/tmp/workloads/<workload_id>
```

**Outputs**: `/data1/huangzhe/code/gpu-simulate-test/tmp/real_runs/<run_id>/`

### 3) Generate Vidur profiling bundle (one-time per model+hardware)

```bash
pixi run vidur-profile \
  model=qwen3_0_6b \
  hardware=a100 \
  profiling.root=/data1/huangzhe/code/gpu-simulate-test/tmp/vidur_profiling/a100/qwen3_0_6b
```

### 4) Run Vidur simulation from repo root

```bash
pixi run vidur-sim \
  model=qwen3_0_6b \
  hardware=a100 \
  profiling.root=/data1/huangzhe/code/gpu-simulate-test/tmp/vidur_profiling/a100/qwen3_0_6b \
  workload.workload_dir=/data1/huangzhe/code/gpu-simulate-test/tmp/workloads/<workload_id>
```

**Outputs**: `/data1/huangzhe/code/gpu-simulate-test/tmp/vidur_runs/<run_id>/`

### 5) Compare one real run vs one sim run

```bash
pixi run compare-runs \
  real_run_dir=/data1/huangzhe/code/gpu-simulate-test/tmp/real_runs/<run_id> \
  sim_run_dir=/data1/huangzhe/code/gpu-simulate-test/tmp/vidur_runs/<run_id>
```

**Outputs**: `/data1/huangzhe/code/gpu-simulate-test/tmp/comparisons/<comparison_id>/summary.md`

## Notes

- All timing fields are integer nanoseconds relative to run start (monotonic).
- If the real run early-stops, comparisons align per-token series using `num_decode_tokens_actual` (sim tokens are truncated per request).

