# Subtask 4.4: Repro integration and interpretation

## Scope

Make host profiling roots easy to use in reproduction runs and make provenance explicit in outputs.

In scope:
- Ensure `paper-fidelity repro` can be run against a host profiling root via Hydra override (`scenario.vidur.profiling_root=<path>`).
- Update report/provenance to clearly capture:
  - which profiling root was used
  - whether it was “paper” (Vidur submodule) or “host” (profiled on this machine)
- Optional: expose a config toggle for CPU overhead modeling (default unchanged).

Out of scope:
- Implementing profiling itself (Subtask 4.3).
- Tests and documentation updates (Subtask 4.5).

## Planned outputs

- Updated report content in `src/gpu_simulate_test/paper_fidelity/report.py` (and/or metadata emitted by `src/gpu_simulate_test/cli/paper_fidelity.py`) to include the effective profiling root and interpretation mode.
- Optional: new config field such as `scenario.vidur.skip_cpu_overhead_modeling` plumbed into `src/gpu_simulate_test/vidur_ext/sim_runner.py` (default remains `True`).

## TODOs

- [X] Job-004-104-001 Add “profiling root” provenance to `results/reports/.../summary.md` and to the written `run_meta.json` (`src/gpu_simulate_test/paper_fidelity/report.py`, `src/gpu_simulate_test/cli/paper_fidelity.py`).
- [X] Job-004-104-002 Define a clear “profiling mode” rule (paper vs host) based on the resolved profiling root path or an explicit config field, and include it in the report.
- [X] Job-004-104-003 Add an optional CPU overhead modeling toggle to configs and plumb into `src/gpu_simulate_test/vidur_ext/sim_runner.py` without changing default behavior.

## Notes

- For “sanity-check reproduction”, the sim-vs-real `% error` can be treated as informational; for “gap reproduction”, it becomes the primary signal. The report should make this interpretation explicit.
