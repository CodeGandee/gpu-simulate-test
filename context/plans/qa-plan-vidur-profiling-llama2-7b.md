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

## Given GPU type and LLM model, what Vidur profiling options exist, and which are configurable in this repo?
> Last revised at: `2026-01-07T09:54:53Z` | Last revised base commit: `bceb0a322dd212499f61074551de6ca126af3219`

- **Vidur compute profilers (upstream flags)**:
  - MLP: `extern/tracked/vidur/vidur/profiling/mlp/main.py` supports `--num_gpus`, `--models`, `--num_tensor_parallel_workers`, `--max_tokens`, `--profile_method`, plus `--disable_ray` and `--output_dir`.
  - Attention: `extern/tracked/vidur/vidur/profiling/attention/main.py` supports `--num_gpus`, `--models`, `--num_tensor_parallel_workers`, `--max_model_len`, `--max_seq_len`, `--min_batch_size`, `--max_batch_size`, `--profile_only_decode|--profile_only_prefill`, `--attention_backend`, `--block_size`, plus `--disable_ray` and `--output_dir`.
  - CPU overhead: `extern/tracked/vidur/vidur/profiling/cpu_overhead/main.py` supports `--models`, `--num_tensor_parallel_workers`, `--max_batch_size`, plus `--output_dir`.
- **Configurable via this repo’s wrapper (`configs/vidur_profiling/bundle.yaml`)**:
  - `profiling.num_gpus` → `--num_gpus` (MLP + attention)
  - `model.model_id` → `--models <hf_id>` (MLP + attention)
  - `profiling.tensor_parallel_size` → `--num_tensor_parallel_workers <tp>` (MLP + attention)
  - `profiling.max_tokens` → `--max_tokens` (MLP) and `--max_model_len/--max_seq_len` (attention)
  - `profiling.cpu_overhead.enabled` → run CPU overhead profiling and stage a Vidur-compatible CSV (default: `false`, to match the paper)
  - `profiling.cpu_overhead.max_batch_size` → `--max_batch_size` (CPU overhead)
  - `profiling.attention.profile_mode` → `--profile_only_decode|--profile_only_prefill|<none>` (attention)
  - `profiling.attention.backend` → `--attention_backend <FLASHINFER|NO_OP|...>` (attention; Sarathi backend)
  - `profiling.attention.block_size` → `--block_size <n>` (attention)
  - `profiling.attention.min_batch_size` / `profiling.attention.max_batch_size` → `--min_batch_size` / `--max_batch_size` (attention)
  - `output.cache_dir` is used as Vidur’s `--output_dir` (where upstream profilers write timestamped outputs); curated CSVs land in the required `output.dir` (CPU overhead is staged as `data/profiling/cpu_overhead/<network_device>/<model>/cpu_overheads.csv`).
- **Not currently configurable via this repo’s wrapper (would require code changes)**:
  - `--disable_ray` (MLP + attention) and MLP `--profile_method` (we always use Vidur defaults in the wrapper).
  - Any changes to dtypes or other hardcoded profiler internals (e.g., attention uses `torch.float16` in `attention/main.py`).
- **Other Vidur profilers (not used by `vidur-profiling` today)**:
  - Network/collectives: `extern/tracked/vidur/vidur/profiling/collectives/main.py` (`--collective`, `--max_collective_size`, `--num_workers_per_node_combinations`, `--output_dir`).

## What values are acceptable for Vidur’s MLP `--profile_method`, and what do they mean?
> Last revised at: `2026-01-07T09:26:28Z` | Last revised base commit: `710c808f79d5c2b6e475d0e541e9c663d48c5627`

- Accepted values are the `ProfileMethod` enum values in `extern/tracked/vidur/vidur/profiling/utils/__init__.py`: `cuda_event`, `kineto`, `perf_counter`, `record_function` (wired into `extern/tracked/vidur/vidur/profiling/mlp/main.py` as `choices=[e.value for e in ProfileMethod]`).
- `cuda_event`: uses CUDA events to time GPU work inside each `CudaTimer` scope (low overhead; GPU-time focused) (`extern/tracked/vidur/vidur/profiling/common/cuda_timer.py`).
- `kineto`: uses `torch.profiler.profile` (Kineto) per `CudaTimer` scope and sums CUDA event times for the scope (highest overhead; useful for detailed profiling/diagnosis) (`extern/tracked/vidur/vidur/profiling/common/cuda_timer.py`).
- `perf_counter`: uses `time.perf_counter()` with `torch.cuda.synchronize()` before/after the timed region (simple wall-clock; includes sync and some CPU overhead) (`extern/tracked/vidur/vidur/profiling/common/cuda_timer.py`).
- `record_function`: captures a full profiler trace and then parses Chrome-trace events to attribute CUDA time to Vidur’s record-function annotations (detailed breakdown; writes JSON traces under `*/profiler_traces/`) (`extern/tracked/vidur/vidur/profiling/utils/record_function_tracer.py`, `extern/tracked/vidur/vidur/profiling/mlp/mlp_wrapper.py`).
- Default in Vidur’s MLP profiler is `record_function` (`extern/tracked/vidur/vidur/profiling/mlp/main.py`). This repo’s `vidur-profiling` wrapper does not currently expose `--profile_method`; it relies on Vidur defaults (see `src/gpu_simulate_test/vidur_ext/profile_runner.py`).

## For CPU overhead profiling, what does it measure, and what do the paper and Vidur repo recommend?
> Last revised at: `2026-01-07T09:54:53Z` | Last revised base commit: `bceb0a322dd212499f61074551de6ca126af3219`

- Vidur’s “CPU overhead” profiling is meant to capture implementation overheads in the serving stack (e.g., scheduling, sampling, detokenization / output processing) on top of pure GPU compute (`extern/tracked/vidur/docs/profiling.md`).
- This repo’s `vidur-profiling` bundle exporter keeps CPU overhead profiling disabled by default (matches the paper); enable it via `profiling.cpu_overhead.enabled=true` in `configs/vidur_profiling/bundle.yaml` (runner: `src/gpu_simulate_test/vidur_ext/profile_runner.py`, patched entrypoint: `src/gpu_simulate_test/vidur_ext/vidur_profiling_cpu_overhead_main.py`).
- In the current upstream implementation, it runs Sarathi’s `LLMEngine` with `scheduler_type="vllm"` and `enable_cpu_op_level_metrics=True`, and reports per-step CPU operation metrics like `SCHEDULE`, `PREPARE_INPUTS_E2E`, `MODEL_EXECUTION_E2E`, `PROCESS_MODEL_OUTPUTS`, and `SAMPLER_E2E` (`extern/tracked/vidur/vidur/profiling/cpu_overhead/benchmark_runner.py`, `extern/tracked/sarathi-serve/sarathi/metrics/constants.py`).
- Each “CPU op” metric is timed via `sarathi.metrics.cpu_timer.CpuTimer`, which uses `time.perf_counter()` and calls `torch.cuda.synchronize()` before recording; interpret these as host wall-clock times for the scoped region (they can include GPU-wait time inside the scope, not just pure CPU compute) (`extern/tracked/sarathi-serve/sarathi/metrics/cpu_timer.py`).
- The CPU-overhead benchmark also computes `ray_comm_time_mean` as the leftover per-step wall time after subtracting the summed tracked CPU op times; this effectively captures un-attributed coordination overhead (e.g., Ray comm/scheduling) (`extern/tracked/vidur/vidur/profiling/cpu_overhead/benchmark_runner.py`).
- Vidur’s repo explicitly recommends profiling CPU overhead “for better fidelity”, but notes it tightly couples the simulator to the specific implementation (e.g., vLLM), and the scripts are “not documented yet” (`extern/tracked/vidur/docs/profiling.md`).
- The Vidur paper states their evaluations use an optimized vLLM fork with CUDA graphs, “which eliminates unnecessary CPU overheads”, and attributes the 7B model’s slightly higher error to higher CPU overhead; to align with paper conditions, keep the serving stack optimized (CUDA graphs / reduced CPU overhead) and consistent with what you simulate, or profile CPU overhead using that same stack (`extern/tracked/vidur/paper/tex/5-eval.tex`).
