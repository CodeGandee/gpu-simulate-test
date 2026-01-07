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

## How are the host profiling artifacts in `results/raw/vidur-profiling/llama2-7b/sarathi-serve/2026-01-07_10-43-39-975600338/` produced, and what do they contain?
> Last revised at: `2026-01-07T13:44:49Z` | Last revised base commit: `1d3073a49749dbd8b7abc86851fd868ccf4982f5`

- Produced by running `pixi run vidur-profiling` (`scripts/run_vidur_profiling_llama2_7b.sh`), which calls `python -m gpu_simulate_test.cli.vidur_profiling_bundle` with a timestamped `output.dir` and defaults from `configs/vidur_profiling/bundle.yaml`.
- The bundle exporter (`src/gpu_simulate_test/vidur_ext/profiling_bundle.py`) runs Vidur’s profilers via `src/gpu_simulate_test/vidur_ext/profile_runner.py`:
  - MLP: `python -m gpu_simulate_test.vidur_ext.vidur_profiling_mlp_main ...`
  - Attention: `python -m gpu_simulate_test.vidur_ext.vidur_profiling_attention_main ...`
  - This run is compute-only (no network profiling, no CPU overhead profiling), with TP=1, `max_tokens=4096`, attention backend `FLASHINFER`, and attention `profile_mode=both` (all recorded in `profiling_meta.json`).
- Curated profiling CSVs (the “useful outputs” for simulation) live under:
  - `data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/mlp.csv` (per-op `time_stats.*` for embedding, layernorm, attention projections, MLP projections/activation, etc., across `num_tokens` for `num_tensor_parallel_workers=1`).
  - `data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention.csv` (per-op `time_stats.*` for attention input reshape / KV cache save / prefill / decode / output reshape, across `(is_prefill, prefill_chunk_size, kv_cache_size, batch_size, ...)` for `num_tensor_parallel_workers=1`).
- Provenance is captured in `profiling_meta.json` (exact commands, resolved Hydra params, git commit/dirty flag, and a small environment snapshot like GPU name and torch version).
- Intermediate/debug outputs are stored under `cache/` (raw Vidur profiler outputs + Hydra logs; may include large traces such as `cache/mlp/*/profiler_traces/*.json`) and are not required once you have the curated CSVs.
- Caveat for sim-vs-real: the profiling grid (and thus `attention.csv` row count) is controlled by Vidur profiler args like `--num_tensor_parallel_workers`, `--max_seq_len`, `--min_batch_size/--max_batch_size`, and `--profile_only_decode/--profile_only_prefill`; this bundle is TP=1 only, so simulations for other TP degrees require re-profiling with matching knobs.

## [question title]
> Last revised at: `2026-01-07T13:41:28Z` | Last revised base commit: `4532f445d0bee8ac33644b44885a4aba671a691d`

- [answer/code]

## [question title]
> Last revised at: `2026-01-07T13:41:28Z` | Last revised base commit: `4532f445d0bee8ac33644b44885a4aba671a691d`

- [answer/code]
