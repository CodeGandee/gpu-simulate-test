# Q&A: 002-sim-vs-real-llama2-7b

## Introduction

This Q&A is for developers (including future maintainers) comparing Vidur simulation timing vs real Sarathi-Serve timing for LLaMA2-7B on this host. Its purpose is to capture runbook-style clarifications (commands, artifact locations, and caveats) when using the paper-fidelity workflow and the host profiling bundle tooling.

**Related docs**
- `specs/002-reproduce-vidur-paper-fidelity/tasks.md`
- `specs/002-reproduce-vidur-paper-fidelity/quickstart.md`
- `specs/002-reproduce-vidur-paper-fidelity/qa-002-reproduce-vidur-paper-fidelity.md`
- `context/plans/plan-vidur-profiling-llama2-7b.md`
- `context/plans/qa-plan-vidur-profiling-llama2-7b.md`
- `results/raw/README.md`

**Key entrypoints and modules**
- `src/gpu_simulate_test/cli/paper_fidelity.py`
- `configs/paper_fidelity/scenario/llama2_7b_arxiv.yaml`
- `configs/paper_fidelity/profile.yaml`
- `src/gpu_simulate_test/paper_fidelity/profiling.py`
- `src/gpu_simulate_test/vidur_ext/profile_runner.py`
- `src/gpu_simulate_test/vidur_ext/sim_runner.py`
- `src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py`
- `configs/vidur_profiling/bundle.yaml`
- `scripts/run_vidur_profiling_llama2_7b.sh`
- `src/sitecustomize.py`

## [question title]
> Last revised at: `2026-01-07T13:41:28Z` | Last revised base commit: `4532f445d0bee8ac33644b44885a4aba671a691d`

- [answer/code]

## [question title]
> Last revised at: `2026-01-07T13:41:28Z` | Last revised base commit: `4532f445d0bee8ac33644b44885a4aba671a691d`

- [answer/code]

## [question title]
> Last revised at: `2026-01-07T13:41:28Z` | Last revised base commit: `4532f445d0bee8ac33644b44885a4aba671a691d`

- [answer/code]
