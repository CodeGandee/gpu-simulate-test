# Smoke checklist (v1): `vidur-cli`

This checklist is intended to be kept **green** (no unchecked boxes) to avoid regressions
in the basic `vidur-cli` workflow.

## Preflight + inspection

- [x] `resources show` fails from a non-repo directory without `GSIM_REPO_ROOT` (actionable error + non-zero exit).
- [x] `resources show` succeeds with `GSIM_REPO_ROOT` set (prints resolved paths + sources).
- [x] `configs list --group model` prints at least one key and a source path.

## Run directory + trace

- [x] `svr init-run ...` prints an absolute `<run_dir>` and writes `run_state.json` + `resources.json`.
- [x] `svr trace --from-lengths ...` writes `trace/trace.csv` + `trace_meta.json` + compatibility CSVs.
- [x] `svr sim --run-dir <run_dir>` fails fast when `profiling_root` is missing and writes `failure.json` (stage=`sim`).

## Full workflow (GPU + model assets required)

Non-gated reference commands (may be slow):

- `svr profile --run-dir <run_dir>`
- `svr sim --run-dir <run_dir>`
- `svr real --run-dir <run_dir>`
- `svr report --run-dir <run_dir>`

