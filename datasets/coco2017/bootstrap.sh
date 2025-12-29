#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REF_DIR="${SCRIPT_DIR}"

require_cmd() {
	for c in "$@"; do
		if ! command -v "$c" >/dev/null 2>&1; then
			echo "missing required command: $c" >&2
			exit 127
		fi
	done
}

require_cmd ln rm

abspath() {
	local p="$1"
	if command -v realpath >/dev/null 2>&1; then
		realpath -m "$p"
	elif command -v python3 >/dev/null 2>&1; then
		python3 - "$p" <<'PY'
import os
import sys

print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
	else
		case "$p" in
			/*) printf '%s\n' "$p" ;;
			*) printf '%s\n' "$(pwd -P)/$p" ;;
		esac
	fi
}

SRC_LINK="${REF_DIR}/source-data"

default_target="/data1/huangzhe/datasets/coco2017"
if [[ -n "${EXTERNAL_REF_ROOT:-}" ]]; then
	target_candidate="${EXTERNAL_REF_ROOT}/coco2017"
else
	target_candidate="${default_target}"
fi

target="$(abspath "${target_candidate}")"

if [[ ! -e "${target}" ]]; then
	echo "Target does not exist: ${target}" >&2
	echo "Set EXTERNAL_REF_ROOT to your local dataset storage root, e.g.:" >&2
	echo "  export EXTERNAL_REF_ROOT=/path/to/datasets" >&2
	exit 1
fi

rm -rf "${SRC_LINK}"
ln -s "${target}" "${SRC_LINK}"

echo "Linked ${SRC_LINK} -> ${target}"
