#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/../../.." && pwd)"

LOG_DIR="${EXPERIMENT_DIR}/logs"
mkdir -p "${LOG_DIR}"

PROMPTS_JSONL="${EXPERIMENT_DIR}/prompts.jsonl"
WORKLOAD_DIR="${EXPERIMENT_DIR}/workload"
REAL_RUN_DIR="${EXPERIMENT_DIR}/real_run"
VIDUR_RUN_DIR="${EXPERIMENT_DIR}/vidur_run"
COMPARE_DIR="${EXPERIMENT_DIR}/compare"

SEED=123
DECODE_TOKENS=64
INTER_ARRIVAL_NS=2000000000 # 2s; intended to avoid overlapping requests in the (sequential) real-bench runner

cat >"${PROMPTS_JSONL}" <<'JSONL'
{"prompt_id":"p0","text":"Write one sentence about GPUs and inference latency."}
{"prompt_id":"p1","text":"Give a short definition of a transformer model."}
{"prompt_id":"p2","text":"Explain the difference between prefill and decode in LLM inference."}
{"prompt_id":"p3","text":"In one paragraph, describe what a latency simulator is used for."}
{"prompt_id":"p4","text":"List three factors that can affect TTFT for an LLM."}
{"prompt_id":"p5","text":"Write a short explanation of batching in LLM serving."}
{"prompt_id":"p6","text":"Describe KV cache in one sentence."}
{"prompt_id":"p7","text":"Explain why profiling data matters for latency simulation."}
JSONL

echo "repo_root=${REPO_ROOT}"
echo "experiment_dir=${EXPERIMENT_DIR}"
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "${REPO_ROOT}"

pixi run workload-spec \
  model=qwen3_0_6b \
  workload.prompts="${PROMPTS_JSONL}" \
  workload.seed="${SEED}" \
  workload.num_decode_tokens="${DECODE_TOKENS}" \
  workload.arrival.kind=fixed_interval \
  workload.arrival.inter_arrival_ns="${INTER_ARRIVAL_NS}" \
  workload.workload_dir="${WORKLOAD_DIR}" \
  hydra.run.dir="${WORKLOAD_DIR}" \
  2>&1 | tee "${LOG_DIR}/workload-spec.log"

CUDA_VISIBLE_DEVICES=0 pixi run real-bench \
  backend=sarathi \
  model=qwen3_0_6b \
  model.model_id="${REPO_ROOT}/models/qwen3-0.6b/source-data" \
  workload.workload_dir="${WORKLOAD_DIR}" \
  hydra.run.dir="${REAL_RUN_DIR}" \
  2>&1 | tee "${LOG_DIR}/real-bench.log"

pixi run vidur-sim \
  model=qwen3_0_6b \
  hardware=a100 \
  vidur.profiling.root="${REPO_ROOT}/tmp/vidur_profiling/a100/qwen3_0_6b" \
  workload.workload_dir="${WORKLOAD_DIR}" \
  hydra.run.dir="${VIDUR_RUN_DIR}" \
  2>&1 | tee "${LOG_DIR}/vidur-sim.log"

pixi run compare-runs \
  real_run_dir="${REAL_RUN_DIR}" \
  sim_run_dir="${VIDUR_RUN_DIR}" \
  hydra.run.dir="${COMPARE_DIR}" \
  2>&1 | tee "${LOG_DIR}/compare-runs.log"

echo "done."
echo "workload_dir=${WORKLOAD_DIR}"
echo "real_run_dir=${REAL_RUN_DIR}"
echo "vidur_run_dir=${VIDUR_RUN_DIR}"
echo "compare_dir=${COMPARE_DIR}"

