#!/usr/bin/env bash
set -euo pipefail

# Pixi runs tasks from the project root. For `vidur-cli`, we want "run-from-anywhere"
# semantics: resolve relative paths (like .vidur-config/default.toml) relative to the
# directory where the user invoked `pixi run ...`, not the repo root.
cd "${INIT_CWD:-$PWD}"

exec python -m gpu_simulate_test.cli.vidur_cli "$@"

