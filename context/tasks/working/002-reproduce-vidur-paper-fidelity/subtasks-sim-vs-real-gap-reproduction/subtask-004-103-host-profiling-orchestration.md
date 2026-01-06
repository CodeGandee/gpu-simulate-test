# Subtask 4.3: Host profiling orchestration

## Scope

Implement the host profiling workflow that produces a Vidur-compatible profiling root, using Vidur’s profiling entrypoints (which depend on Sarathi modules).

In scope:
- Run GPU microbenchmarks for the current scenario’s model:
  - MLP profiling → `mlp.csv`
  - Attention profiling → `attention.csv`
- Assemble a profiling root with the layout Vidur expects:
  - `data/profiling/compute/<device>/<model_id>/{mlp,attention}.csv`
  - Optionally stage `data/profiling/network/<network_device>/*.csv` for TP/PP > 1 scenarios.
- Write provenance metadata near the profiling root (git commit, env snapshot, timestamps, and the resolved scenario parameters).

Out of scope:
- Changing how Vidur simulation computes metrics (Subtask 4.4 for integration knobs).
- Test coverage and docs (Subtask 4.5).

## Planned outputs

- New module `src/gpu_simulate_test/paper_fidelity/profiling.py` that:
  - Takes a scenario config (model/device/network_device) and produces a profiling root under `tmp/paper_fidelity/profiling_roots/...`
  - Calls into `src/gpu_simulate_test/vidur_ext/profile_runner.py` to run profiling and stage required CSVs
  - Writes `profiling_meta.json` (or similar) next to the profiling root
- Updates to `src/gpu_simulate_test/vidur_ext/profile_runner.py` as needed to support:
  - configurable staging output location under `tmp/paper_fidelity/profiling_outputs/...`
  - scenario-driven device/network device settings

## TODOs

- [X] Job-004-103-001 Implement `src/gpu_simulate_test/paper_fidelity/profiling.py` orchestration function (input: scenario config; output: profiling root path).
- [X] Job-004-103-002 Extend `src/gpu_simulate_test/vidur_ext/profile_runner.py` to accept an explicit staging/output directory (avoid `_staging` under the final profiling root).
- [X] Job-004-103-003 Ensure the orchestrator writes provenance JSON including `build_env_snapshot()` and `get_git_info()` outputs (reuse `src/gpu_simulate_test/io/provenance.py`).
- [X] Job-004-103-004 Fail fast with clear errors when CUDA/Sarathi prerequisites are missing (consistent with existing paper-fidelity runners).

## Notes

- Vidur profiling scripts import Sarathi modules (`sarathi.model_executor.*`, `sarathi.metrics.*`), so this subtask implicitly depends on `extern/tracked/sarathi-serve` being initialized and available in the Pixi env.
