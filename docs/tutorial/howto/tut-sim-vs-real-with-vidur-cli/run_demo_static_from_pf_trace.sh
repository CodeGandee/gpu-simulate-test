#!/usr/bin/env bash
# End-to-end demo runner for `vidur-cli` sim-vs-real (STATIC case) using:
# - Model: llama2_7b
# - Hardware preset: a100
# - Real backend: sarathi
# - Trace: this directory's tracked `inputs/trace_import.csv`
#
# This creates a fresh `<run_dir>` under a fresh workspace (defaults to `<repo>/tmp/...`),
# then runs the full pipeline:
#   init-run → trace(import) → profile → sim → real → report
#
# Usage (from repo root):
#   docs/tutorial/howto/tut-sim-vs-real-with-vidur-cli/run_demo_static_from_pf_trace.sh
#
# Optional: also write a sanitized report snapshot under the workspace (does NOT modify this tutorial dir):
#   docs/tutorial/howto/tut-sim-vs-real-with-vidur-cli/run_demo_static_from_pf_trace.sh --snapshot-report

# Strict mode:
# -e: fail fast
# -u: treat unset vars as errors
# -o pipefail: fail if any command in a pipeline fails
set -euo pipefail

# Absolute directory of this script (so the script works regardless of current working dir).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine repo root. Prefer git (fast + correct when inside a repo),
# and fall back to walking up until we find `pyproject.toml`.
REPO_ROOT="$(
  git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true
)"
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$SCRIPT_DIR"
  while [[ "$REPO_ROOT" != "/" && ! -f "$REPO_ROOT/pyproject.toml" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
  done
  if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
    echo "could not determine repo root (expected pyproject.toml above $SCRIPT_DIR)" >&2
    exit 1
  fi
fi

# Optional flag to write a sanitized report snapshot under the workspace.
SNAPSHOT_REPORT="0"
if [[ "${1:-}" == "--snapshot-report" ]]; then
  SNAPSHOT_REPORT="1"
  shift
fi
if [[ "${1:-}" != "" ]]; then
  echo "usage: $0 [--snapshot-report]" >&2
  exit 2
fi

# This repo's CLI runner expects GSIM_REPO_ROOT so it can resolve config + resource roots.
export GSIM_REPO_ROOT="${GSIM_REPO_ROOT:-$REPO_ROOT}"

# Keep the attention-profiling compatibility patch disabled in this driver script.
#
# The profiling runner enables it only for the attention profiling subprocess to avoid interfering
# with other stages (Sarathi replay, CPU overhead profiling).
export GPU_SIMULATE_TEST_ENABLE_VIDUR_ATTENTION_COMPAT="0"

# GPU pinning:
# - This host reserves GPUs 4,5 for these experiments (see repo `.env`).
# - `GSIM_CUDA_VISIBLE_DEVICES` is used by repo code; `CUDA_VISIBLE_DEVICES` is used by CUDA/PyTorch.
export GSIM_CUDA_VISIBLE_DEVICES="${GSIM_CUDA_VISIBLE_DEVICES:-4,5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GSIM_CUDA_VISIBLE_DEVICES}"

# Workspace dir: where `vidur-cli` will create per-run directories.
# We intentionally use `<repo>/tmp/<exp>` (gitignored) so it is easy to delete and doesn't pollute the repo.
EXP_NAME="${EXP_NAME:-vidur_cli_demo_pf_trace_llama2_7b_static_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
export GSIM_VIDUR_WORKSPACE_DIR="${GSIM_VIDUR_WORKSPACE_DIR:-$REPO_ROOT/tmp/$EXP_NAME}"
mkdir -p "$GSIM_VIDUR_WORKSPACE_DIR"

# Safety: never write any outputs under this tutorial directory.
SCRIPT_DIR_REAL="$(readlink -f "$SCRIPT_DIR" || true)"
WORKSPACE_REAL="$(readlink -f "$GSIM_VIDUR_WORKSPACE_DIR" || true)"
if [[ -n "$SCRIPT_DIR_REAL" && -n "$WORKSPACE_REAL" && "$WORKSPACE_REAL" == "$SCRIPT_DIR_REAL"* ]]; then
  echo "refusing to use a workspace under the tutorial dir (would overwrite tracked files):" >&2
  echo "  tutorial_dir=$SCRIPT_DIR_REAL" >&2
  echo "  workspace_dir=$WORKSPACE_REAL" >&2
  exit 1
fi

# This demo vendors a deterministic trace import file (static → arrival_time_ns should be 0).
# Copy it into the workspace so this script never risks modifying tracked inputs.
TRACE_IMPORT_CSV_SRC="${SCRIPT_DIR}/inputs/trace_import.csv"
if [[ ! -f "$TRACE_IMPORT_CSV_SRC" ]]; then
  echo "missing trace import CSV: $TRACE_IMPORT_CSV_SRC" >&2
  exit 1
fi
WORKSPACE_INPUTS_DIR="$GSIM_VIDUR_WORKSPACE_DIR/inputs"
mkdir -p "$WORKSPACE_INPUTS_DIR"
TRACE_IMPORT_CSV="$WORKSPACE_INPUTS_DIR/trace_import.csv"
cp -a "$TRACE_IMPORT_CSV_SRC" "$TRACE_IMPORT_CSV"

# Submodules are required for Vidur + Sarathi.
if [[ ! -d "$REPO_ROOT/extern/tracked/vidur" || ! -d "$REPO_ROOT/extern/tracked/sarathi-serve" ]]; then
  echo "missing submodules (vidur/sarathi); run:" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi

# Model/tokenizer assets must exist. The tutorial uses `models/llama2-7b-hf/bootstrap.sh` to populate this.
if [[ ! -e "$REPO_ROOT/models/llama2-7b-hf/source-data" ]]; then
  echo "missing model ref: $REPO_ROOT/models/llama2-7b-hf/source-data" >&2
  echo "hint: run: bash models/llama2-7b-hf/bootstrap.sh" >&2
  exit 1
fi

# Print key environment to make reproduction/debugging easier.
echo "GSIM_REPO_ROOT=$GSIM_REPO_ROOT"
echo "GSIM_VIDUR_WORKSPACE_DIR=$GSIM_VIDUR_WORKSPACE_DIR"
echo "GSIM_CUDA_VISIBLE_DEVICES=$GSIM_CUDA_VISIBLE_DEVICES"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "GPU_SIMULATE_TEST_ENABLE_VIDUR_ATTENTION_COMPAT=$GPU_SIMULATE_TEST_ENABLE_VIDUR_ATTENTION_COMPAT"

# MLP profiling method selection is REQUIRED (no hidden defaults).
#
# This tutorial defaults to `record_function`:
# - historically Vidur defaulted to record_function
# - this repo fixed a record_function gap for CUDA *driver*-launched kernels (005-vidur-mlp-cuda-driver)
#
# Alternatives: `cuda_event` | `record_function_org` | `kineto` | `perf_counter`. See:
# - docs/tutorial/in-depth/adv-tut-vidur-cli-mlp-profile-methods/
export GSIM_VIDUR_MLP_PROFILE_METHOD="${GSIM_VIDUR_MLP_PROFILE_METHOD:-record_function}"

# Attention profiling decode batch coverage (parity-critical):
# - `backend.scheduler.max_num_seqs` caps the real run's batch size (default 16 in this tutorial).
# - `profiling.attention.max_batch_size` must cover that cap to avoid extrapolating attention timing.
#   Lower this (e.g. 1) only if you accept the fidelity loss for faster profiling.
export GSIM_VIDUR_ATTENTION_MAX_BATCH_SIZE="${GSIM_VIDUR_ATTENTION_MAX_BATCH_SIZE:-16}"

# MLP validation mode for staged `mlp.csv`.
export GSIM_VIDUR_MLP_VALIDATION_MODE="${GSIM_VIDUR_MLP_VALIDATION_MODE:-strict}"

# Optional: automatically retry with a fallback method when validation fails.
export GSIM_VIDUR_MLP_FALLBACK_ENABLED="${GSIM_VIDUR_MLP_FALLBACK_ENABLED:-false}"
export GSIM_VIDUR_MLP_FALLBACK_METHOD="${GSIM_VIDUR_MLP_FALLBACK_METHOD:-cuda_event}"

# Missing-value handling for staged `mlp.csv`:
# - auto: strict => reject; non_strict => drop (per-target) during consumption.
# - reject: always fail on NaNs.
# - drop: allow NaNs (consumers must handle them).
# - zero: allow NaNs (consumers fill missing targets with 0.0 per target).
export GSIM_VIDUR_MLP_NAN_POLICY="${GSIM_VIDUR_MLP_NAN_POLICY:-zero}"

# Missing-value handling when consuming a profiling root in `svr sim`.
export GSIM_VIDUR_SIM_MLP_VALIDATION_MODE="${GSIM_VIDUR_SIM_MLP_VALIDATION_MODE:-strict}"
export GSIM_VIDUR_SIM_MLP_NAN_POLICY="${GSIM_VIDUR_SIM_MLP_NAN_POLICY:-zero}"

# CPU overhead profiling improves sim-vs-real parity, but can be flaky on some hosts
# (Ray worker startup, CUDA init issues, OOM at large batch sizes).
#
# This tutorial defaults to enabling it. Disable explicitly if you are debugging compute-only behavior
# or if CPU overhead profiling is failing on your machine.
export GSIM_VIDUR_INCLUDE_CPU_OVERHEAD="${GSIM_VIDUR_INCLUDE_CPU_OVERHEAD:-true}"

CPU_OVERHEAD_ENABLED="false"
case "$(echo "$GSIM_VIDUR_INCLUDE_CPU_OVERHEAD" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) CPU_OVERHEAD_ENABLED="true" ;;
esac

# 1) Create a fresh run directory with the chosen presets.
RUN_DIR="$(
  pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr init-run \
    model=llama2_7b hardware=a100 backend=sarathi workload=default vidur=default
)"
echo "RUN_DIR=$RUN_DIR"

# 2) Import the trace into `<run_dir>/trace/trace.csv` (canonical schema: arrival_time_ns).
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr trace --run-dir "$RUN_DIR" --import-trace "$TRACE_IMPORT_CSV"

# 3) Profile on THIS host, to generate profiling data used by the simulator (GPU kernels + CPU overhead).
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr profile --run-dir "$RUN_DIR" \
  "profiling.cpu_overhead.enabled=${CPU_OVERHEAD_ENABLED}" \
  "profiling.attention.max_batch_size=${GSIM_VIDUR_ATTENTION_MAX_BATCH_SIZE}" \
  "profiling.mlp.profile_method=${GSIM_VIDUR_MLP_PROFILE_METHOD}" \
  "profiling.mlp.validation.mode=${GSIM_VIDUR_MLP_VALIDATION_MODE}" \
  "profiling.mlp.fallback.enabled=${GSIM_VIDUR_MLP_FALLBACK_ENABLED}" \
  "profiling.mlp.fallback.method=${GSIM_VIDUR_MLP_FALLBACK_METHOD}" \
  "profiling.mlp.validation.nan_policy=${GSIM_VIDUR_MLP_NAN_POLICY}"

# 4) Run the Vidur simulation using the imported trace + the freshly generated profiling bundle.
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr sim --run-dir "$RUN_DIR" \
  "vidur.validation.mlp.mode=${GSIM_VIDUR_SIM_MLP_VALIDATION_MODE}" \
  "vidur.validation.mlp.nan_policy=${GSIM_VIDUR_SIM_MLP_NAN_POLICY}"

# 5) Run the real replay (Sarathi) using the SAME trace, so sim-vs-real is comparable.
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr real --run-dir "$RUN_DIR"

# 6) Generate the report and print the stable output path.
# The report also snapshots the scored CSVs under `<run_dir>/report/inputs/` so the report is self-contained.
SUMMARY_MD="$(pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr report --run-dir "$RUN_DIR")"
echo "SUMMARY_MD=$SUMMARY_MD"

if [[ "$SNAPSHOT_REPORT" == "1" ]]; then
  # Write a sanitized report snapshot under the workspace (never into this tutorial directory).
  SNAPSHOT_DIR="$GSIM_VIDUR_WORKSPACE_DIR/report_snapshot_$(basename "$RUN_DIR")"
  if [[ -e "$SNAPSHOT_DIR" ]]; then
    echo "snapshot dir already exists (refusing to overwrite): $SNAPSHOT_DIR" >&2
    exit 1
  fi
  mkdir -p "$SNAPSHOT_DIR"
  cp -a "$RUN_DIR/report/." "$SNAPSHOT_DIR/"

  # Sanitize machine-local absolute paths in the snapshot.
  MODEL_REF="$(readlink -f "$REPO_ROOT/models/llama2-7b-hf/source-data" || true)"
  SANITIZER_PY="${SCRIPT_DIR}/scripts/sanitize_expected_report.py"
  if [[ ! -f "$SANITIZER_PY" ]]; then
    echo "missing sanitizer script: $SANITIZER_PY" >&2
    exit 1
  fi
  pixi run python "$SANITIZER_PY" \
    --expected-dir "$SNAPSHOT_DIR" \
    --run-dir "$RUN_DIR" \
    --repo-root "$REPO_ROOT" \
    --model-ref "$MODEL_REF"
  echo "wrote report snapshot: $SNAPSHOT_DIR"
fi
