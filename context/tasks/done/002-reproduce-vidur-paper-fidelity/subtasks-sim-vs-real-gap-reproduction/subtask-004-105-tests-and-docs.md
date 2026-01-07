# Subtask 4.5: Validation and docs

## Scope

Add a minimal validation story and document how to run host-calibrated gap reproduction.

In scope:
- Unit tests for profiling-root assembly and required file validation.
- A bounded GPU/manual smoke test for host profiling.
- Documentation updates to the feature spec docs.

Out of scope:
- Changing scoring semantics (keep existing pass/warn/fail unless explicitly requested).

## Planned outputs

- `tests/unit/test_paper_fidelity_profiling_root.py` (new)
- `tests/manual/test_paper_fidelity_profile_smoke.py` (new)
- Updates to:
  - `specs/002-reproduce-vidur-paper-fidelity/tasks.md`
  - `specs/002-reproduce-vidur-paper-fidelity/quickstart.md`

## TODOs

- [X] Job-004-105-001 Add unit tests covering profiling-root assembly and validation via `src/gpu_simulate_test/vidur_ext/profiling_root.py` (use small fixture CSVs under `tests/fixtures/`).
- [X] Job-004-105-002 Add a manual smoke script that runs `paper-fidelity profile` for a single scenario and verifies the expected CSVs exist (writes under `tmp/` only).
- [X] Job-004-105-003 Update `specs/002-reproduce-vidur-paper-fidelity/tasks.md` to include host-gap-reproduction tasks and their validation steps.
- [X] Job-004-105-004 Update `specs/002-reproduce-vidur-paper-fidelity/quickstart.md` with commands for both “sanity-check” and “sim-vs-real gap” reproduction.

## Notes

- Keep the GPU smoke test bounded (single model, TP1/PP1) to reduce runtime and resource use.
