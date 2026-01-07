# Subtask 4.1: Profile CLI + config scaffold

## Scope

Add a first-class CLI entrypoint for host profiling and a Hydra config scaffold.

In scope:
- `paper-fidelity profile --scenario <name>` command surface and Hydra wiring.
- A new config file under `configs/paper_fidelity/` for profiling runs.

Out of scope:
- Actually running profiling/microbenchmarking logic (handled in Subtask 4.3).
- Any behavior changes to existing `paper-fidelity {repro,trace,score}` commands.

## Planned outputs

- `configs/paper_fidelity/profile.yaml`
- Updated CLI dispatcher in `src/gpu_simulate_test/cli/paper_fidelity.py` with a `profile` subcommand
- A Hydra entrypoint that prints the created profiling root path (even if the underlying profiling logic is stubbed until Subtask 4.3 is complete)

## TODOs

- [X] Job-004-101-001 Create `configs/paper_fidelity/profile.yaml` (mirrors other paper-fidelity configs; accepts `scenario=<name>` and profiling knobs).
- [X] Job-004-101-002 Add `profile` subcommand to the argparse wrapper in `src/gpu_simulate_test/cli/paper_fidelity.py` and forward Hydra overrides.
- [X] Job-004-101-003 Add a Hydra main `_profile_main()` in `src/gpu_simulate_test/cli/paper_fidelity.py` that invokes the profiling orchestrator and prints the profiling root path.

## Notes

- Keep the CLI behavior consistent with existing commands: print the output directory (profiling root) to stdout.
- Heavy artifacts must go under `tmp/` (profiling outputs and profiling roots); nothing should be written under tracked source directories.
