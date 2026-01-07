#!/usr/bin/env bash
set -euo pipefail

# Host profiling bundle for Vidur compute microbenchmarks (LLaMA2-7B).
#
# Outputs:
# - Profiling root (useful outputs): results/raw/vidur-profiling/llama2-7b/sarathi-serve/<run_id>/
# - Cache/intermediates (debugging): <output-dir>/cache/

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="$(date -u +%Y-%m-%d_%H-%M-%S-%N)"

output_dir="${repo_root}/results/raw/vidur-profiling/llama2-7b/sarathi-serve/${run_id}"

python -m gpu_simulate_test.cli.vidur_profiling_bundle \
  "output.dir=${output_dir}" \
  "output.model_slug=llama2-7b" \
  "output.scheduler_name=sarathi-serve"

