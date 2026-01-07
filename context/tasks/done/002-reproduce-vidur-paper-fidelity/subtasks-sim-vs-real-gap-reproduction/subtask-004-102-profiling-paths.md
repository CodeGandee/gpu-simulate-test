# Subtask 4.2: Profiling artifact paths and layouts

## Scope

Define and implement stable path conventions for host profiling artifacts under `tmp/paper_fidelity/`.

In scope:
- Add path helpers for:
  - raw profiling outputs (large, intermediate)
  - finalized host profiling roots (Vidur-compatible layout)
- Ensure all paths are deterministic and scenario-scoped.

Out of scope:
- Running profiling (Subtask 4.3).
- Updating reports (Subtask 4.4).

## Planned outputs

- Updated `src/gpu_simulate_test/paper_fidelity/paths.py` with helpers such as:
  - `profiling_outputs_dir(<scenario>, <run_id/timestamp>)`
  - `profiling_root_dir(<scenario>, <run_id/timestamp>)`
- Clear directory layout documented in code (and later in `specs/.../quickstart.md` via Subtask 4.5).

## TODOs

- [X] Job-004-102-001 Extend `src/gpu_simulate_test/paper_fidelity/paths.py` to include `tmp/paper_fidelity/profiling_outputs/...` and `tmp/paper_fidelity/profiling_roots/...` helpers.
- [X] Job-004-102-002 Ensure all generated paths are absolute in run metadata (resolve against `paths.repo_root` when needed).
- [X] Job-004-102-003 Add a small helper for writing/locating profiling provenance JSON alongside the profiling root (schema can be finalized in Subtask 4.3).

## Notes

- Keep the directory names consistent with existing `tmp/paper_fidelity/{traces,runs}/...` conventions.
