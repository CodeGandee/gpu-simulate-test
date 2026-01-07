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

## How do we use the host profiling bundle to run a Vidur simulation for LLaMA2-7B on this host?
> Last revised at: `2026-01-07T13:51:47Z` | Last revised base commit: `122e75940eb4f2ce02913dba85dd0990bf55f702`

- Use the profiling root as `scenario.vidur.profiling_root` and run the paper-fidelity pipeline (it invokes Vidur under the hood):
  - Static: `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload static scenario.vidur.profiling_root=results/raw/vidur-profiling/llama2-7b/sarathi-serve/2026-01-07_10-43-39-975600338`
  - Dynamic: `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic scenario.vidur.profiling_root=results/raw/vidur-profiling/llama2-7b/sarathi-serve/2026-01-07_10-43-39-975600338`
- Vidur simulation uses `src/gpu_simulate_test/vidur_ext/sim_runner.py` to:
  - Validate required profiling inputs exist (always `mlp.csv` + `attention.csv`; network/CPU overhead only when TP/PP or CPU modeling is enabled) (`src/gpu_simulate_test/vidur_ext/profiling_root.py`).
  - Configure Vidur’s predictor to read from `<profiling_root>/data/profiling/...` via absolute paths in `RandomForrestExecutionTimePredictorConfig` (so the simulator does not depend on `cwd`).
- Outputs to compare against real timing are written under `tmp/paper_fidelity/runs/llama2_7b_arxiv/sim/`:
  - `request_metrics.csv` (paper-fidelity schema, derived from Vidur’s `vidur_raw/request_metrics.csv`)
  - `run_meta.json` and `vidur_raw/` (Vidur’s native outputs + `config.json`).
- Constraints for this particular profiling bundle:
  - It is compute-only and TP=1/PP=1 calibrated, so keep `scenario.vidur.tensor_parallel_size=1` and `scenario.vidur.num_pipeline_stages=1` unless you also provide the required network profiling CSVs.
  - CPU overhead modeling is skipped by default (`scenario.vidur.skip_cpu_overhead_modeling=true`); if you enable CPU overhead modeling, you must also provide `data/profiling/cpu_overhead/<network_device>/<model>/cpu_overheads.csv`.

## When running `paper-fidelity repro --workload dynamic`, who determines request arrival gaps (inter-arrival times), and how?
> Last revised at: `2026-01-07T14:01:28Z` | Last revised base commit: `369ec6bafbd5d8e1b47fd959a46cafc59fb20efd`

- The **trace generator** determines arrival gaps by writing `arrived_at` (seconds since start) into the canonical `trace.csv`; Vidur and Sarathi both consume the same `trace.csv`.
- For dynamic workloads, arrivals are generated by `src/gpu_simulate_test/paper_fidelity/traces.py:add_poisson_arrivals()`:
  - It samples **exponential inter-arrival times** with mean `1/qps` (Poisson process), using `workload.seed` for determinism (`configs/paper_fidelity/workload/dynamic.yaml`).
  - It then sets `arrived_at = [0.0, cumsum(inter_arrivals)...]`.
- For the baseline `llama2_7b_arxiv` scenario, the dynamic pipeline first performs **capacity discovery** and sets QPS to `qps_85`:
  - `src/gpu_simulate_test/cli/paper_fidelity.py` runs Sarathi at candidate QPS values, scores overload using P99(`request_scheduling_delay`) > threshold (default 5s), and computes `qps_85 = 0.85 * capacity_qps` (`specs/002-reproduce-vidur-paper-fidelity/tasks.md`).
  - The final dynamic `trace.csv` is regenerated using that `qps_85` and `workload.seed`.
- In the Sarathi replay loop, the runner submits requests when wall-clock time reaches `start + arrived_at` (`src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py`); Vidur reads the same `arrived_at` values via `TraceRequestGeneratorConfig(trace_file=...)` (`src/gpu_simulate_test/vidur_ext/sim_runner.py`).
- Where `trace.csv` is written (assuming the run output dir is `<output-dir>`):
  - Capacity search uses an intermediate trace at `tmp/paper_fidelity/runs/<scenario>/capacity/trace.csv` (overwritten as QPS candidates change).
  - The final dynamic trace used by both Vidur and Sarathi is written to `tmp/paper_fidelity/traces/<scenario>/trace.csv` (with `trace_meta.json` alongside).
  - `<output-dir>` (Hydra run dir) does not control these paths; it only controls Hydra’s working directory (`configs/paper_fidelity/repro.yaml`).
