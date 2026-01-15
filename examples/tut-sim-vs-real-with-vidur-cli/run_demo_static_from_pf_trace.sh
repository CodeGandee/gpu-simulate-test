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
  export EXPECTED_DIR RUN_DIR REPO_ROOT MODEL_REF
  pixi run python - <<'PY'
import json
import os
from pathlib import Path

expected_dir = Path(os.environ["EXPECTED_DIR"]).resolve()
run_dir = os.environ.get("RUN_DIR", "")
repo_root = os.environ.get("REPO_ROOT", "")
model_ref = os.environ.get("MODEL_REF", "")

def _replace_str(s: str) -> str:
    if run_dir and s.startswith(run_dir):
        return "<RUN_DIR>" + s[len(run_dir):]
    if repo_root and s.startswith(repo_root):
        return "<REPO_ROOT>" + s[len(repo_root):]
    if model_ref and s.startswith(model_ref):
        return "<MODEL_REF>" + s[len(model_ref):]
    return s

def _walk(x):
    if isinstance(x, str):
        return _replace_str(x)
    if isinstance(x, list):
        return [_walk(v) for v in x]
    if isinstance(x, dict):
        return {k: _walk(v) for k, v in x.items()}
    return x

summary_md = expected_dir / "summary.md"
if summary_md.exists():
    text = summary_md.read_text(encoding="utf-8")
    for src, dst in [(run_dir, "<RUN_DIR>"), (repo_root, "<REPO_ROOT>"), (model_ref, "<MODEL_REF>")]:
        if src:
            text = text.replace(src, dst)
    # Make the report title stable across refreshes.
    lines = text.splitlines()
    if lines and lines[0].startswith("# Sim-vs-Real Report:"):
        lines[0] = "# Sim-vs-Real Report: <RUN_TAG>"
    summary_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

run_meta = expected_dir / "run_meta.json"
if run_meta.exists():
    meta = json.loads(run_meta.read_text(encoding="utf-8"))
    meta = _walk(meta)
    # Make the report title stable across refreshes (best-effort: keep the config, drop only the run tag).
    if isinstance(meta, dict) and "run_dir" in meta:
        meta["run_dir"] = "<RUN_DIR>"
    run_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  echo "refreshed expected_report/: $EXPECTED_DIR"
fi
