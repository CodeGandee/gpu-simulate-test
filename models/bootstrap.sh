#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

echo "Bootstrapping model references in ${SCRIPT_DIR} ..."

bash "${SCRIPT_DIR}/qwen3-0.6b/bootstrap.sh"

echo "Done."

