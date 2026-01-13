#!/usr/bin/env bash
set -euo pipefail

# Batch runner: paper-fidelity sweeps across scenarios/workloads at a fixed scale.
#
# This replaces the removed `paper-fidelity matrix` subcommand. Use this script when you want to:
# - run host profiling (`paper-fidelity profile`) per scenario
# - run sim-vs-real repro (`paper-fidelity repro`) for static and/or dynamic
# - apply global TP/PP overrides consistently across all cases
# - record a simple, append-only sweep log (`cases.jsonl`) for later aggregation
#
# Outputs:
# - Per-case reports: `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<scenario_tag>/...`
# - Sweep log: `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/sweep_<run_id>/cases.jsonl`
#
# Prereqs:
# - `git submodule update --init --recursive`
# - `pixi install`
# - GPU pinning configured via `GSIM_CUDA_VISIBLE_DEVICES` (repo `.env` or exported)

usage() {
  cat <<'EOF'
Usage: scripts/paper_fidelity_sweep.sh [options]

Options:
  --run-id <id>                  Sweep identifier (default: UTC timestamp)
  --date <UTC-YYYY-MM-DD>        Report date directory (default: today in UTC)
  --scale <small|medium|full>    Scale for all runs (default: small)
  --scenarios <csv>              Scenario keys (default: internlm_20b_arxiv,llama2_70b_arxiv,qwen_72b_arxiv)
  --workloads <csv>              Workloads (default: static,dynamic)

  --tp <n>                       Global tensor parallel size for all cases (default: 1)
  --pp <n>                       Global pipeline parallel size for all cases (default: 1)

  --include-cpu-overhead         Include CPU overhead microbenchmarks in profiling (default)
  --no-include-cpu-overhead      Disable CPU overhead microbenchmarks in profiling

  --stop-on-failure              Stop after the first failed action
  -h, --help                     Show this help
EOF
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "${s}"
}

file_safe_slug() {
  local raw="$1"
  local slug
  slug="$(printf '%s' "${raw}" | sed -E 's/[^A-Za-z0-9_.-]+/_/g')"
  slug="$(trim "${slug}")"
  if [[ -z "${slug}" ]]; then
    slug="unknown"
  fi
  printf '%s' "${slug}"
}

last_non_empty_line() {
  local text="$1"
  local line
  local last=""
  while IFS= read -r line; do
    if [[ -n "$(trim "${line}")" ]]; then
      last="${line}"
    fi
  done <<<"${text}"
  printf '%s' "${last}"
}

json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\t'/\\t}"
  printf '%s' "${s}"
}

append_case_jsonl() {
  local out="$1"
  local json="$2"
  printf '%s\n' "${json}" >> "${out}"
}

require_cmd() {
  for c in "$@"; do
    command -v "${c}" >/dev/null 2>&1 || { echo "missing: ${c}" 1>&2; exit 127; }
  done
}

run_id=""
date_utc=""
scale="small"
scenarios_csv=""
workloads_csv="static,dynamic"
tp="1"
pp="1"
include_cpu_overhead=1
stop_on_failure=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) run_id="$2"; shift 2 ;;
    --date) date_utc="$2"; shift 2 ;;
    --scale) scale="$2"; shift 2 ;;
    --scenarios) scenarios_csv="$2"; shift 2 ;;
    --workloads) workloads_csv="$2"; shift 2 ;;
    --tp) tp="$2"; shift 2 ;;
    --pp) pp="$2"; shift 2 ;;
    --include-cpu-overhead) include_cpu_overhead=1; shift ;;
    --no-include-cpu-overhead) include_cpu_overhead=0; shift ;;
    --stop-on-failure) stop_on_failure=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" 1>&2; usage; exit 2 ;;
  esac
done

require_cmd sed date mktemp pixi

if [[ -z "${date_utc}" ]]; then
  date_utc="$(date -u +%Y-%m-%d)"
fi

if [[ -z "${run_id}" ]]; then
  run_id="$(date -u +%Y-%m-%d_%H-%M-%S-%N)"
fi
run_id_slug="$(file_safe_slug "${run_id}")"

case "${scale}" in
  small|medium|full) ;;
  *) echo "Invalid --scale: ${scale} (expected small|medium|full)" 1>&2; exit 2 ;;
esac

if [[ ! "${tp}" =~ ^[0-9]+$ ]] || [[ "${tp}" -lt 1 ]]; then
  echo "Invalid --tp: ${tp} (expected integer >= 1)" 1>&2
  exit 2
fi
if [[ ! "${pp}" =~ ^[0-9]+$ ]] || [[ "${pp}" -lt 1 ]]; then
  echo "Invalid --pp: ${pp} (expected integer >= 1)" 1>&2
  exit 2
fi

num_gpus="$((tp * pp))"

if [[ -z "${scenarios_csv}" ]]; then
  scenarios_csv="internlm_20b_arxiv,llama2_70b_arxiv,qwen_72b_arxiv"
fi

IFS=',' read -r -a scenarios_raw <<<"${scenarios_csv}"
scenarios=()
for s in "${scenarios_raw[@]}"; do
  s="$(trim "${s}")"
  [[ -n "${s}" ]] && scenarios+=("${s}")
done
if [[ "${#scenarios[@]}" -eq 0 ]]; then
  echo "--scenarios resolved to an empty list" 1>&2
  exit 2
fi

IFS=',' read -r -a workloads_raw <<<"${workloads_csv}"
workloads=()
for w in "${workloads_raw[@]}"; do
  w="$(trim "${w}")"
  [[ -n "${w}" ]] && workloads+=("${w}")
done
if [[ "${#workloads[@]}" -eq 0 ]]; then
  echo "--workloads resolved to an empty list" 1>&2
  exit 2
fi
for w in "${workloads[@]}"; do
  case "${w}" in
    static|dynamic) ;;
    *) echo "Invalid workload: ${w} (expected static|dynamic)" 1>&2; exit 2 ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

sweep_dir="${repo_root}/results/reports/${date_utc}/paper_fidelity/sweep_${run_id_slug}"
mkdir -p "${sweep_dir}"
cases_jsonl="${sweep_dir}/cases.jsonl"

echo "Sweep: run_id=${run_id_slug} date=${date_utc} scale=${scale} tp=${tp} pp=${pp} num_gpus=${num_gpus}"
echo "Sweep log: ${cases_jsonl}"

for scenario_key in "${scenarios[@]}"; do
  scenario_tag="${scenario_key}_sweep_${run_id_slug}"

  echo "== Scenario: ${scenario_key} (scenario.name=${scenario_tag}) =="

  profile_cmd=(pixi run paper-fidelity profile --scenario "${scenario_key}")
  if [[ "${include_cpu_overhead}" -eq 1 ]]; then
    profile_cmd+=(--include-cpu-overhead)
  else
    profile_cmd+=(--no-include-cpu-overhead)
  fi
  profile_cmd+=(
    "scenario.name=${scenario_tag}"
    "scenario.real.parallel.tensor_parallel_size=${tp}"
    "scenario.real.parallel.pipeline_parallel_size=${pp}"
    "scenario.vidur.tensor_parallel_size=${tp}"
    "scenario.vidur.num_pipeline_stages=${pp}"
    "profiling.tensor_parallel_size=${tp}"
    "profiling.num_gpus=${num_gpus}"
  )

  profile_stdout_file="$(mktemp)"
  profile_stderr_file="$(mktemp)"
  set +e
  "${profile_cmd[@]}" >"${profile_stdout_file}" 2>"${profile_stderr_file}"
  profile_rc=$?
  set -e
  profile_stdout="$(cat "${profile_stdout_file}")"
  profile_stderr="$(cat "${profile_stderr_file}")"
  rm -f "${profile_stdout_file}" "${profile_stderr_file}"

  profile_out_line="$(last_non_empty_line "${profile_stdout}")"
  if [[ "${profile_rc}" -ne 0 ]]; then
    err_line="$(last_non_empty_line "${profile_stderr}")"
    if [[ -z "${err_line}" ]]; then
      err_line="profile failed (exit=${profile_rc})"
    fi

    append_case_jsonl "${cases_jsonl}" "$(
      printf '{\"action\":\"profile\",\"scenario_key\":\"%s\",\"scenario_name\":\"%s\",\"scale\":\"%s\",\"tp\":%s,\"pp\":%s,\"status\":\"failure\",\"output\":\"%s\",\"error\":\"%s\"}' \
        "$(json_escape "${scenario_key}")" \
        "$(json_escape "${scenario_tag}")" \
        "$(json_escape "${scale}")" \
        "$(json_escape "${tp}")" \
        "$(json_escape "${pp}")" \
        "$(json_escape "${profile_out_line}")" \
        "$(json_escape "${err_line}")"
    )"

    echo "Profile failed: ${err_line}" 1>&2
    if [[ "${stop_on_failure}" -eq 1 ]]; then
      exit 1
    fi
    continue
  fi

  profiling_root="${profile_out_line}"
  if [[ -z "${profiling_root}" ]]; then
    append_case_jsonl "${cases_jsonl}" "$(
      printf '{\"action\":\"profile\",\"scenario_key\":\"%s\",\"scenario_name\":\"%s\",\"scale\":\"%s\",\"tp\":%s,\"pp\":%s,\"status\":\"failure\",\"output\":\"\",\"error\":\"%s\"}' \
        "$(json_escape "${scenario_key}")" \
        "$(json_escape "${scenario_tag}")" \
        "$(json_escape "${scale}")" \
        "$(json_escape "${tp}")" \
        "$(json_escape "${pp}")" \
        "$(json_escape "profile produced no output")"
    )"
    echo "Profile produced no output; cannot continue scenario=${scenario_key}" 1>&2
    if [[ "${stop_on_failure}" -eq 1 ]]; then
      exit 1
    fi
    continue
  fi

  append_case_jsonl "${cases_jsonl}" "$(
    printf '{\"action\":\"profile\",\"scenario_key\":\"%s\",\"scenario_name\":\"%s\",\"scale\":\"%s\",\"tp\":%s,\"pp\":%s,\"status\":\"success\",\"profiling_root\":\"%s\"}' \
      "$(json_escape "${scenario_key}")" \
      "$(json_escape "${scenario_tag}")" \
      "$(json_escape "${scale}")" \
      "$(json_escape "${tp}")" \
      "$(json_escape "${pp}")" \
      "$(json_escape "${profiling_root}")"
  )"

  for workload in "${workloads[@]}"; do
    echo "-- Repro: workload=${workload} scale=${scale}"

    report_folder="${scenario_tag}"
    if [[ "${workload}" != "static" ]]; then
      report_folder="${scenario_tag}_${workload}_${scale}"
    fi
    expected_report_dir="${repo_root}/results/reports/${date_utc}/paper_fidelity/${report_folder}"

    repro_cmd=(
      pixi run paper-fidelity repro
      --scenario "${scenario_key}"
      --workload "${workload}"
      --scale "${scale}"
      "scenario.name=${scenario_tag}"
      "scenario.vidur.profiling_root=${profiling_root}"
      "scenario.real.parallel.tensor_parallel_size=${tp}"
      "scenario.real.parallel.pipeline_parallel_size=${pp}"
      "scenario.vidur.tensor_parallel_size=${tp}"
      "scenario.vidur.num_pipeline_stages=${pp}"
    )

    repro_stdout_file="$(mktemp)"
    repro_stderr_file="$(mktemp)"
    set +e
    "${repro_cmd[@]}" >"${repro_stdout_file}" 2>"${repro_stderr_file}"
    repro_rc=$?
    set -e
    repro_stdout="$(cat "${repro_stdout_file}")"
    repro_stderr="$(cat "${repro_stderr_file}")"
    rm -f "${repro_stdout_file}" "${repro_stderr_file}"

    repro_out_line="$(last_non_empty_line "${repro_stdout}")"
    if [[ "${repro_rc}" -ne 0 ]]; then
      err_line="$(last_non_empty_line "${repro_stderr}")"
      if [[ -z "${err_line}" ]]; then
        err_line="repro failed (exit=${repro_rc})"
      fi

      failure_record_json="${expected_report_dir}/failure_record.json"
      if [[ -f "${failure_record_json}" ]]; then
        repro_out_line="${failure_record_json}"
      fi

      append_case_jsonl "${cases_jsonl}" "$(
        printf '{\"action\":\"repro\",\"scenario_key\":\"%s\",\"scenario_name\":\"%s\",\"workload\":\"%s\",\"scale\":\"%s\",\"tp\":%s,\"pp\":%s,\"status\":\"failure\",\"output\":\"%s\",\"error\":\"%s\"}' \
          "$(json_escape "${scenario_key}")" \
          "$(json_escape "${scenario_tag}")" \
          "$(json_escape "${workload}")" \
          "$(json_escape "${scale}")" \
          "$(json_escape "${tp}")" \
          "$(json_escape "${pp}")" \
          "$(json_escape "${repro_out_line}")" \
          "$(json_escape "${err_line}")"
      )"

      echo "Repro failed: ${err_line}" 1>&2
      if [[ "${stop_on_failure}" -eq 1 ]]; then
        exit 1
      fi
      continue
    fi

    report_dir="${expected_report_dir}"
    if [[ ! -d "${report_dir}" ]]; then
      report_dir="${repro_out_line}"
    fi
    append_case_jsonl "${cases_jsonl}" "$(
      printf '{\"action\":\"repro\",\"scenario_key\":\"%s\",\"scenario_name\":\"%s\",\"workload\":\"%s\",\"scale\":\"%s\",\"tp\":%s,\"pp\":%s,\"status\":\"success\",\"report_dir\":\"%s\"}' \
        "$(json_escape "${scenario_key}")" \
        "$(json_escape "${scenario_tag}")" \
        "$(json_escape "${workload}")" \
        "$(json_escape "${scale}")" \
        "$(json_escape "${tp}")" \
        "$(json_escape "${pp}")" \
        "$(json_escape "${report_dir}")"
    )"
  done
done

echo "Done. Sweep log: ${cases_jsonl}"
