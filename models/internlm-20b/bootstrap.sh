#!/usr/bin/env bash
set -euo pipefail

require_cmd() {
	for c in "$@"; do
		command -v "$c" >/dev/null 2>&1 || { echo "missing: $c" >&2; exit 127; }
	done
}

require_cmd yq realpath ln mkdir rm

ASSUME_YES=false
CLEAN_ONLY=false
while [[ $# -gt 0 ]]; do
	case "$1" in
	-y|--yes) ASSUME_YES=true; shift ;;
	--clean) CLEAN_ONLY=true; shift ;;
	*) echo "unknown arg: $1" >&2; exit 2 ;;
	esac
done

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CFG="${SCRIPT_DIR}/bootstrap.yaml"

ROOT_ENV="$(yq -r '.env.model_root_env' "${CFG}")"
DEFAULT_ROOT="$(yq -r '.env.default_model_root' "${CFG}")"
SUBDIR="$(yq -r '.model.source_subdir' "${CFG}")"
LINK_NAME="$(yq -r '.model.repo_link_name' "${CFG}")"

set +u; BASE="${!ROOT_ENV-}"; set -u
if [[ -z "${BASE}" && -n "${EXTERNAL_REF_ROOT:-}" ]]; then
	BASE="${EXTERNAL_REF_ROOT}"
	echo "Note: using EXTERNAL_REF_ROOT (legacy) as model root; prefer ${ROOT_ENV}." >&2
fi
BASE="${BASE:-${DEFAULT_ROOT}}"

TARGET="$(realpath -m "${BASE}/${SUBDIR}")"
LINK_PATH="${SCRIPT_DIR}/${LINK_NAME}"

if "${CLEAN_ONLY}"; then
	echo "Cleaning model link: ${LINK_PATH}"
	if [[ -L "${LINK_PATH}" ]]; then
		rm -f -- "${LINK_PATH}"
	elif [[ -e "${LINK_PATH}" ]]; then
		echo "exists but not a symlink: ${LINK_PATH}" >&2
	else
		echo "already clean"
	fi
	exit 0
fi

echo "Model bootstrap: ${LINK_PATH} -> ${TARGET}"
echo "  Base (${ROOT_ENV}): ${BASE}"
mkdir -p -- "$(dirname "${LINK_PATH}")"

if [[ ! -e "${TARGET}" ]]; then
	echo "Target does not exist: ${TARGET}" >&2
	echo "Set ${ROOT_ENV} to your local model storage root, e.g.:" >&2
	echo "  export ${ROOT_ENV}=/path/to/llm-models" >&2
	exit 1
fi

if [[ -e "${LINK_PATH}" || -L "${LINK_PATH}" ]]; then
	if "${ASSUME_YES}"; then
		rm -rf -- "${LINK_PATH}"
	else
		read -r -p "Replace existing link/path at ${LINK_PATH}? [y/N]: " a
		[[ "${a,,}" == y* ]] || exit 0
		rm -rf -- "${LINK_PATH}"
	fi
fi

ln -s -- "${TARGET}" "${LINK_PATH}"
echo "Linked ${LINK_PATH} -> ${TARGET}"

if yq -e '.model.required_files' "${CFG}" >/dev/null 2>&1; then
	mapfile -t REQ < <(yq -r '.model.required_files[]' "${CFG}")
	miss=()
	for f in "${REQ[@]}"; do
		[[ -e "${TARGET}/${f}" ]] || miss+=("${f}")
	done
	if ((${#miss[@]})); then
		echo "Warning: missing in target: ${miss[*]}" >&2
	else
		echo "OK: required files present"
	fi
fi
