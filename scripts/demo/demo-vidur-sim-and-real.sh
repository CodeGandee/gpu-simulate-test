#!/usr/bin/env bash
export GSIM_CUDA_VISIBLE_DEVICES=0
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_DISABLE_MEMORY_MONITOR=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
set -euo pipefail

# Detect pixi
PIXI_CMD="pixi"
if ! command -v pixi &> /dev/null; then
    if [ -f "/root/.pixi/bin/pixi" ]; then
        PIXI_CMD="/root/.pixi/bin/pixi"
    else
        echo "Error: pixi not found in PATH or /root/.pixi/bin/pixi"
        exit 1
    fi
fi

# Find the latest profiling root
# Assumes structure: tmp/paper_fidelity/profiling_roots/llama2_7b_arxiv/<timestamp>
PROFILING_ROOT=$(find tmp/paper_fidelity/profiling_roots/llama2_7b_arxiv -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)

if [ -z "$PROFILING_ROOT" ]; then
    echo "Warning: No profiling root found in tmp/paper_fidelity/profiling_roots/llama2_7b_arxiv/"
    echo "Using default paper-provided profiling bundle (sanity check mode)."
    PROFILE_ARG=""
else
    echo "Using profiling root: $PROFILING_ROOT"
    PROFILE_ARG="scenario.vidur.profiling_root=${PROFILING_ROOT}"
fi

# Run Static (Small Scale)
 echo "----------------------------------------------------------------"
 echo "Running Static Workload (Small Scale)..."
 echo "----------------------------------------------------------------"
$PIXI_CMD run paper-fidelity repro \
  --scenario llama2_7b_arxiv \
  --workload static \
  --scale small \
  $PROFILE_ARG

# Run Dynamic (Small Scale)
 echo "----------------------------------------------------------------"
 echo "Running Dynamic Workload (Small Scale)..."
 echo "----------------------------------------------------------------"
$PIXI_CMD run paper-fidelity repro \
  --scenario llama2_7b_arxiv \
  --workload dynamic \
  --scale small \
  $PROFILE_ARG

# Organize Output to tmp/
 echo "----------------------------------------------------------------"
 echo "Organizing results to tmp/results..."
mkdir -p tmp/results
LATEST_REPORT_DIR=$(find results/reports -type d -name "llama2_7b_arxiv*" | sort | tail -n 1)
if [ -n "$LATEST_REPORT_DIR" ]; then
    cp -r "$LATEST_REPORT_DIR" tmp/results/
    echo "Copied latest report to tmp/results/$(basename "$LATEST_REPORT_DIR")"
else
    echo "Warning: No report directory found to copy."
fi