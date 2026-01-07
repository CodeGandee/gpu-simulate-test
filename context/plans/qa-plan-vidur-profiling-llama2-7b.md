# Q&A: plan-vidur-profiling-llama2-7b

## Introduction

This Q&A captures implementation questions for the Vidur host-profiling bundle workflow (LLaMA2-7B), intended for developers (including future maintainers) operating or extending the profiling/export pipeline.

**Related docs**
- `context/plans/plan-vidur-profiling-llama2-7b.md`
- `context/summaries/vidur-kb/about-vendor-provided-data.md`
- `context/summaries/vidur-kb/about-vidur-gpu-simulator.md`

**Key entrypoints and modules**
- `configs/vidur_profiling/bundle.yaml`
- `src/gpu_simulate_test/cli/vidur_profiling_bundle.py`
- `src/gpu_simulate_test/vidur_ext/profiling_bundle.py`
- `src/gpu_simulate_test/vidur_ext/profile_runner.py`
- `tests/manual/test_vidur_profiling_bundle_smoke.py`
- `pyproject.toml`

## How do we do compute profiling on this host for the LLaMA2-7B model?
> Last revised at: `2026-01-07T08:36:33Z` | Last revised base commit: `4059227c22da9789a275dcd9e8ef1063520c3ed5`

- Run the curated bundle exporter: `pixi run vidur-profiling` (defaults are driven by `configs/vidur_profiling/bundle.yaml`, but `output.dir` is supplied by `scripts/run_vidur_profiling_llama2_7b.sh`).
- It executes Vidur’s MLP + attention microbenchmarks on the local CUDA GPU via `src/gpu_simulate_test/vidur_ext/profile_runner.py` (wrapping `vidur.profiling.mlp.main` and `vidur.profiling.attention.main`).
- Outputs (curated profiling root) land under the required `output.dir` and include `data/profiling/compute/...` and `profiling_meta.json` with the exact commands (`src/gpu_simulate_test/vidur_ext/profiling_bundle.py`).
- Intermediate/debug outputs land under `output.cache_dir` (defaults to `<output.dir>/cache`) (`configs/vidur_profiling/bundle.yaml`).
- Useful bounded run (explicit output dir): `pixi run python -m gpu_simulate_test.cli.vidur_profiling_bundle output.dir=tmp/vidur_profiling_bundle_smoke/run1 profiling.max_tokens=256 profiling.attention.max_batch_size=1 profiling.attention.profile_mode=both`.
- If you need to force knobs: override `model.model_id=meta-llama/Llama-2-7b-hf`, `hardware.hardware_id=<device>`, and attention settings like `profiling.attention.backend=FLASHINFER` in the same command line.

## [question title]
> Last revised at: `2026-01-07T08:11:45Z` | Last revised base commit: `4059227c22da9789a275dcd9e8ef1063520c3ed5`

- [answer/code]

## [question title]
> Last revised at: `2026-01-07T08:11:45Z` | Last revised base commit: `4059227c22da9789a275dcd9e8ef1063520c3ed5`

- [answer/code]
