#!/usr/bin/env bash
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

# Run profiling with CPU overhead
echo "Running paper-fidelity profile..."
$PIXI_CMD run paper-fidelity profile --scenario llama2_7b_arxiv --include-cpu-overhead
