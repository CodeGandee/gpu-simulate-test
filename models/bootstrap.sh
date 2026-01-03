#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CFG="${SCRIPT_DIR}/bootstrap.yaml"

echo "Bootstrapping model references in ${SCRIPT_DIR} ..."

failures=0

bootstrap_ref() {
	local ref="$1"
	shift
	if bash "${SCRIPT_DIR}/${ref}/bootstrap.sh" "$@"; then
		return 0
	fi
	echo "Warning: failed to bootstrap ${ref} (missing local files?)." >&2
	failures=$((failures + 1))
	return 0
}

require_cmd() {
	for c in "$@"; do
		command -v "$c" >/dev/null 2>&1 || { echo "missing: $c" >&2; exit 127; }
	done
}

require_cmd yq

mapfile -t REFS < <(yq -r '.models.refs[]' "${CFG}")
for ref in "${REFS[@]}"; do
	bootstrap_ref "${ref}" "$@"
done

if [[ "${failures}" -ne 0 ]]; then
	echo "Completed with ${failures} warning(s). See messages above for missing references." >&2
fi

echo "Done."
