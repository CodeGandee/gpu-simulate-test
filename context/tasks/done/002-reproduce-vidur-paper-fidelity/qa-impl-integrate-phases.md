# Q&A: impl-integrate-phases

## Introduction

This Q&A captures implementation questions that came up while integrating the phases of `002-reproduce-vidur-paper-fidelity`. It is intended for developers (including future maintainers) working on the end-to-end `paper-fidelity` workflow.

**Related docs**
- `context/tasks/done/002-reproduce-vidur-paper-fidelity/impl-integrate-phases.md`
- `context/tasks/done/002-reproduce-vidur-paper-fidelity/qa-impl-phase-3-repro-report.md`
- `specs/002-reproduce-vidur-paper-fidelity/tasks.md`
- `specs/002-reproduce-vidur-paper-fidelity/quickstart.md`
- `context/summaries/vidur-kb/about-vendor-provided-data.md`

**Key entrypoints and modules**
- `src/gpu_simulate_test/cli/paper_fidelity.py`
- `src/gpu_simulate_test/paper_fidelity/profiling.py`
- `src/gpu_simulate_test/vidur_ext/profile_runner.py`
- `configs/paper_fidelity/profile.yaml`
- `src/gpu_simulate_test/vidur_ext/sim_runner.py`

## Do we have implemented profiling tools to get microbenchmark results (like those in `extern/tracked/vidur/data/profiling/`) using actual hardware on this host?
> Last revised at: `2026-01-07T11:13:15Z` | Last revised base commit: `67fa08b5a545a035889b020fdcd823a04342b6a6`

- Yes: Phase 9 (“Host profiling”) is implemented in `specs/002-reproduce-vidur-paper-fidelity/tasks.md` and exposed via `pixi run paper-fidelity profile --scenario <name>` (`src/gpu_simulate_test/cli/paper_fidelity.py`).
- The command runs Vidur’s profilers on the current machine and writes a Vidur-compatible profiling root under `tmp/paper_fidelity/profiling_roots/<scenario>/<run_id>/data/profiling/...` (`src/gpu_simulate_test/paper_fidelity/profiling.py`, `src/gpu_simulate_test/vidur_ext/profile_runner.py`).
- What it profiles on-host by default: **compute** microbenchmarks (**MLP** + **attention**) for the selected model/hardware (`src/gpu_simulate_test/vidur_ext/profile_runner.py` runs `gpu_simulate_test.vidur_ext.vidur_profiling_mlp_main` and `gpu_simulate_test.vidur_ext.vidur_profiling_attention_main`).
- Network/collectives are **not** microbenchmarked by the host-profiling workflow; if needed, network CSVs can be sourced from `extern/tracked/vidur/data/profiling/network/<network_device>/...` (or produced by Vidur’s collectives profiler separately).
- CPU overhead profiling is **supported** by the shared runner (`src/gpu_simulate_test/vidur_ext/profile_runner.py`) but is **disabled by default** to match the paper’s “optimized serving stack” conditions; it is currently exposed via the dedicated `pixi run vidur-profiling` bundle exporter (`configs/vidur_profiling/bundle.yaml`, `profiling.cpu_overhead.enabled=true`) rather than `paper-fidelity profile`.
- If attention profiling fails, the runner can fall back to a packaged `attention.csv` template when enabled (still producing a runnable profiling root, but it is no longer a pure host-measured bundle) (`src/gpu_simulate_test/vidur_ext/profile_runner.py`).
