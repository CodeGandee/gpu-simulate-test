#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

echo "Bootstrapping dataset references in ${SCRIPT_DIR} ..."

bash "${SCRIPT_DIR}/coco2017/bootstrap.sh"

echo "Done."

