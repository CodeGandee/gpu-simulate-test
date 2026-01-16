#!/usr/bin/env bash
set -euo pipefail

# Sweep Vidur MLP profiling methods for the `vidur-cli` static sim-vs-real tutorial.
#
# Runs 4 end-to-end pipelines (one per MLP profile_method):
#   init-run → trace(import) → profile → sim → real → report
#
# Outputs are written under a fresh directory in `<repo>/tmp/`.
#
# Usage:
#   docs/tutorial/in-depth/adv-tut-vidur-cli-mlp-profile-methods/run_sweep_static_profile_methods.sh
#
# Maintainers:
#   docs/tutorial/in-depth/adv-tut-vidur-cli-mlp-profile-methods/run_sweep_static_profile_methods.sh --snapshot-expected

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(
  git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true
)"
if [[ -z "$REPO_ROOT" ]]; then
  echo "could not determine repo root via git (run from inside the repo)" >&2
  exit 1
fi

SNAPSHOT_EXPECTED="0"
if [[ "${1:-}" == "--snapshot-expected" ]]; then
  SNAPSHOT_EXPECTED="1"
  shift
fi
if [[ "${1:-}" != "" ]]; then
  echo "usage: $0 [--snapshot-expected]" >&2
  exit 2
fi

export GSIM_REPO_ROOT="${GSIM_REPO_ROOT:-$REPO_ROOT}"

export GSIM_CUDA_VISIBLE_DEVICES="${GSIM_CUDA_VISIBLE_DEVICES:-4,5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GSIM_CUDA_VISIBLE_DEVICES}"

SWEEP_NAME="${SWEEP_NAME:-vidur_cli_mlp_profile_method_sweep_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
SWEEP_DIR="${SWEEP_DIR:-$REPO_ROOT/tmp/$SWEEP_NAME}"
mkdir -p "$SWEEP_DIR"

SCRIPT_DIR_REAL="$(readlink -f "$SCRIPT_DIR" || true)"
SWEEP_DIR_REAL="$(readlink -f "$SWEEP_DIR" || true)"
if [[ -n "$SCRIPT_DIR_REAL" && -n "$SWEEP_DIR_REAL" && "$SWEEP_DIR_REAL" == "$SCRIPT_DIR_REAL"* ]]; then
  echo "refusing to use a sweep dir under the tutorial dir (would overwrite tracked files):" >&2
  echo "  tutorial_dir=$SCRIPT_DIR_REAL" >&2
  echo "  sweep_dir=$SWEEP_DIR_REAL" >&2
  exit 1
fi

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

# Copy the trace import CSV into the sweep dir (never touch tracked tutorial inputs).
TRACE_IMPORT_CSV_SRC="${SCRIPT_DIR}/inputs/trace_import.csv"
if [[ ! -f "$TRACE_IMPORT_CSV_SRC" ]]; then
  echo "missing trace import CSV: $TRACE_IMPORT_CSV_SRC" >&2
  exit 1
fi
mkdir -p "$SWEEP_DIR/inputs"
TRACE_IMPORT_CSV="$SWEEP_DIR/inputs/trace_import.csv"
cp -a "$TRACE_IMPORT_CSV_SRC" "$TRACE_IMPORT_CSV"

echo "GSIM_REPO_ROOT=$GSIM_REPO_ROOT"
echo "GSIM_CUDA_VISIBLE_DEVICES=$GSIM_CUDA_VISIBLE_DEVICES"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SWEEP_DIR=$SWEEP_DIR"

METHODS=(cuda_event record_function kineto perf_counter)

for METHOD in "${METHODS[@]}"; do
  echo ""
  echo "=== Running profile_method=$METHOD ==="

  # Ray + Vidur profiling uses a Singleton TimerStatsStore; make sure we don't reuse a previous session.
  pixi run ray stop --force >/dev/null 2>&1 || true

  METHOD_WORKSPACE_DIR="$SWEEP_DIR/$METHOD"
  mkdir -p "$METHOD_WORKSPACE_DIR"
  export GSIM_VIDUR_WORKSPACE_DIR="$METHOD_WORKSPACE_DIR"

  LOG="$SWEEP_DIR/${METHOD}.log"

  # Ensure the attention-compat patch is NOT enabled for this tutorial.
  # (Some compat patch paths can interact badly with CPU overhead profiling.)
  {
    RUN_DIR="$(
      pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr init-run \
        model=llama2_7b hardware=a100 backend=sarathi workload=default vidur=default
    )"
    echo "RUN_DIR=$RUN_DIR"

    pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr trace --run-dir "$RUN_DIR" --import-trace "$TRACE_IMPORT_CSV"

    pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr profile --run-dir "$RUN_DIR" \
      "profiling.mlp.profile_method=${METHOD}" \
      "profiling.mlp.fallback.enabled=false" \
      "profiling.mlp.fallback.method=cuda_event"

    pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr sim --run-dir "$RUN_DIR"
    pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr real --run-dir "$RUN_DIR"

    SUMMARY_MD="$(pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr report --run-dir "$RUN_DIR")"
    echo "SUMMARY_MD=$SUMMARY_MD"
  } 2>&1 | env -u GPU_SIMULATE_TEST_ENABLE_VIDUR_ATTENTION_COMPAT tee "$LOG"

  RUN_DIR_CAPTURE="$(rg "^RUN_DIR=" "$LOG" | tail -n1 | cut -d= -f2-)"
  SUMMARY_MD_CAPTURE="$(rg "^SUMMARY_MD=" "$LOG" | tail -n1 | cut -d= -f2-)"

  printf '%s\n' "$RUN_DIR_CAPTURE" > "$SWEEP_DIR/${METHOD}_run_dir.txt"
  printf '%s\n' "$SUMMARY_MD_CAPTURE" > "$SWEEP_DIR/${METHOD}_summary_path.txt"
  cp -a "$SUMMARY_MD_CAPTURE" "$SWEEP_DIR/${METHOD}_summary.md"
  cp -a "$(dirname "$SUMMARY_MD_CAPTURE")/run_meta.json" "$SWEEP_DIR/${METHOD}_report_run_meta.json" || true
  cp -a "$(dirname "$SUMMARY_MD_CAPTURE")/scores.json" "$SWEEP_DIR/${METHOD}_scores.json" || true
done

pixi run python "$SCRIPT_DIR/scripts/summarize_sweep.py" --sweep-dir "$SWEEP_DIR"

if [[ "$SNAPSHOT_EXPECTED" == "1" ]]; then
  SNAPSHOT_DIR="$SWEEP_DIR/expected_snapshot"
  if [[ -e "$SNAPSHOT_DIR" ]]; then
    echo "snapshot dir already exists (refusing to overwrite): $SNAPSHOT_DIR" >&2
    exit 1
  fi
  pixi run python "$SCRIPT_DIR/scripts/sanitize_expected_outputs.py" \
    --sweep-dir "$SWEEP_DIR" \
    --expected-dir "$SNAPSHOT_DIR"
  echo "wrote expected snapshot: $SNAPSHOT_DIR"
fi

echo ""
echo "done: $SWEEP_DIR"
echo "comparison: $SWEEP_DIR/comparison.md"

