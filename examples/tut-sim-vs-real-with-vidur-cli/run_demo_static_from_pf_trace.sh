#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REFRESH_EXPECTED_REPORT="0"
if [[ "${1:-}" == "--refresh-expected-report" ]]; then
  REFRESH_EXPECTED_REPORT="1"
  shift
fi
if [[ "${1:-}" != "" ]]; then
  echo "usage: $0 [--refresh-expected-report]" >&2
  exit 2
fi

export GSIM_REPO_ROOT="${GSIM_REPO_ROOT:-$REPO_ROOT}"

# GPU pinning: this host reserves GPUs 4,5 for these experiments (see repo .env).
export GSIM_CUDA_VISIBLE_DEVICES="${GSIM_CUDA_VISIBLE_DEVICES:-4,5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GSIM_CUDA_VISIBLE_DEVICES}"

EXP_NAME="${EXP_NAME:-vidur_cli_demo_pf_trace_llama2_7b_static_$(date -u +%Y%m%dT%H%M%SZ)}"
export GSIM_VIDUR_WORKSPACE_DIR="${GSIM_VIDUR_WORKSPACE_DIR:-$REPO_ROOT/tmp/$EXP_NAME}"
mkdir -p "$GSIM_VIDUR_WORKSPACE_DIR"

TRACE_IMPORT_CSV="${SCRIPT_DIR}/inputs/trace_import.csv"
if [[ ! -f "$TRACE_IMPORT_CSV" ]]; then
  echo "missing trace import CSV: $TRACE_IMPORT_CSV" >&2
  exit 1
fi

if [[ ! -d "$REPO_ROOT/extern/tracked/vidur" || ! -d "$REPO_ROOT/extern/tracked/sarathi-serve" ]]; then
  echo "missing submodules (vidur/sarathi); run:" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi

if [[ ! -e "$REPO_ROOT/models/llama2-7b-hf/source-data" ]]; then
  echo "missing model ref: $REPO_ROOT/models/llama2-7b-hf/source-data" >&2
  echo "hint: run: bash models/llama2-7b-hf/bootstrap.sh" >&2
  exit 1
fi

echo "GSIM_REPO_ROOT=$GSIM_REPO_ROOT"
echo "GSIM_VIDUR_WORKSPACE_DIR=$GSIM_VIDUR_WORKSPACE_DIR"
echo "GSIM_CUDA_VISIBLE_DEVICES=$GSIM_CUDA_VISIBLE_DEVICES"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

RUN_DIR="$(
  pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr init-run \
    model=llama2_7b hardware=a100 backend=sarathi workload=default vidur=default
)"
echo "RUN_DIR=$RUN_DIR"

pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr trace --run-dir "$RUN_DIR" --import-trace "$TRACE_IMPORT_CSV"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr profile --run-dir "$RUN_DIR"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr sim --run-dir "$RUN_DIR"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr real --run-dir "$RUN_DIR"

SUMMARY_MD="$(pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr report --run-dir "$RUN_DIR")"
echo "SUMMARY_MD=$SUMMARY_MD"

if [[ "$REFRESH_EXPECTED_REPORT" == "1" ]]; then
  EXPECTED_DIR="${SCRIPT_DIR}/expected_report"
  rm -rf "$EXPECTED_DIR"
  mkdir -p "$EXPECTED_DIR"
  cp -a "$RUN_DIR/report/." "$EXPECTED_DIR/"
  # Sanitize machine-local absolute paths in the tracked snapshot.
  MODEL_REF="$(readlink -f "$REPO_ROOT/models/llama2-7b-hf/source-data" || true)"
  SANITIZER_PY="${SCRIPT_DIR}/scripts/sanitize_expected_report.py"
  if [[ ! -f "$SANITIZER_PY" ]]; then
    echo "missing sanitizer script: $SANITIZER_PY" >&2
    exit 1
  fi
  pixi run python "$SANITIZER_PY" \
    --expected-dir "$EXPECTED_DIR" \
    --run-dir "$RUN_DIR" \
    --repo-root "$REPO_ROOT" \
    --model-ref "$MODEL_REF"
  echo "refreshed expected_report/: $EXPECTED_DIR"
fi
