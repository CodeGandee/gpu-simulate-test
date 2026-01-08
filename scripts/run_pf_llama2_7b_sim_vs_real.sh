#!/usr/bin/env bash
set -euo pipefail

# Batch runner: paper-fidelity sim vs real for LLaMA2-7B (static + dynamic, small/medium/full).
#
# Prereqs (see `context/instructions/prep-dev-env.md`):
# - `git submodule update --init --recursive`
# - `pixi install`
# - CUDA available: `pixi run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
#
# This script records a manifest JSON under:
# - `results/reports/<date>/paper_fidelity/<base_scenario>_sim_vs_real_<run_id>/manifest.json`

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

date_utc="$(date -u +%Y-%m-%d)"
run_id="$(date -u +%Y-%m-%d_%H-%M-%S-%N)"

base_scenario="${PF_SCENARIO:-llama2_7b_arxiv}"
scenario_tag="${base_scenario}_sim_vs_real_${run_id}"

profiling_root="${PF_PROFILING_ROOT:-${repo_root}/results/raw/vidur-profiling/llama2-7b/sarathi-serve/latest}"

if [[ ! -d "${profiling_root}/data/profiling" ]]; then
  cat <<EOF 1>&2
Missing Vidur profiling root:
  ${profiling_root}

Generate it first (host microbenchmark bundle), then re-run this script:
  pixi run vidur-profiling
EOF
  exit 1
fi

manifest_dir="${repo_root}/results/reports/${date_utc}/paper_fidelity/${scenario_tag}"
mkdir -p "${manifest_dir}"

runs_tsv="${manifest_dir}/runs.tsv"
printf "workload\tscale\tscenario_name\treport_dir\n" > "${runs_tsv}"

workloads=(static dynamic)
scales=(small medium full)

for workload in "${workloads[@]}"; do
  for scale in "${scales[@]}"; do
    scenario_name="${scenario_tag}_${workload}_${scale}"
    echo "Running workload=${workload} scale=${scale} (scenario.name=${scenario_name})"

    report_dir="$(
      pixi run paper-fidelity repro \
        --scenario "${base_scenario}" \
        --workload "${workload}" \
        --scale "${scale}" \
        "scenario.name=${scenario_name}" \
        "scenario.vidur.profiling_root=${profiling_root}" \
        | tail -n 1
    )"

    printf "%s\t%s\t%s\t%s\n" "${workload}" "${scale}" "${scenario_name}" "${report_dir}" >> "${runs_tsv}"
  done
done

manifest_json="${manifest_dir}/manifest.json"
pixi run python -m gpu_simulate_test.paper_fidelity.manifest \
  --runs-tsv "${runs_tsv}" \
  --out "${manifest_json}" \
  --base-scenario "${base_scenario}" \
  --run-id "${run_id}" \
  --repo-root "${repo_root}"

echo "Wrote manifest: ${manifest_json}"

