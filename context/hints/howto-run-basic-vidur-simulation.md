# How to run a basic Vidur simulation

## HEADER
- **Purpose**: Quick steps and example commands for running a basic LLM inference simulation with Vidur (Microsoft’s inference system simulator).
- **Status**: Active
- **Date**: 2025-12-29
- **Dependencies**: `extern/tracked/vidur` submodule; Python environment compatible with Vidur; optional Weights & Biases (wandb) account.
- **Target**: Developers and AI assistants working on this repo.

## What Vidur simulates (at a glance)

Vidur is a high-fidelity LLM inference system simulator that predicts end-to-end behavior (latency/throughput, batching, preemption/restarts, etc.) using profiling data (compute + network) plus a workload generator (synthetic or trace-based).

Key ideas:
- You typically do a one-time profiling phase on real GPUs for new model/SKU combos; once profiling data exists, you can run simulations on CPU-only machines.
- A simulation run is driven by CLI flags that specify the model, device/SKU, parallelism, scheduler, and workload generator.

## Basic “does it run?” simulation

From the Vidur repo root (`extern/tracked/vidur/`):

```sh
python -m vidur.main -h
python -m vidur.main
```

The first command lists all available parameters; the second runs with Vidur defaults.

## A minimal explicit simulation (synthetic workload)

This is a small, explicit run that pins the key knobs (device/model/topology/workload). Adjust values as needed:

```sh
python -m vidur.main \
  --replica_config_device a100 \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --cluster_config_num_replicas 1 \
  --replica_config_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests 64
```

Notes:
- `--replica_config_device` selects the compute profiling folder (for example `data/profiling/compute/a100/...`).
- `--replica_config_model_name` must correspond to a model Vidur has profiling/config support for.
- `TP` (`--replica_config_tensor_parallel_size`) and `PP` (`--replica_config_num_pipeline_stages`) define the parallelism configuration being simulated.

## Trace-driven workload (realistic request lengths + Poisson arrivals)

Vidur can generate request lengths from a trace file and inter-arrivals from a Poisson process:

```sh
python -m vidur.main \
  --replica_config_device a100 \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --cluster_config_num_replicas 1 \
  --replica_config_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests 512 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --trace_request_length_generator_config_max_tokens 16384 \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 6.45 \
  --replica_scheduler_config_type sarathi \
  --sarathi_scheduler_config_batch_size_cap 512 \
  --sarathi_scheduler_config_chunk_size 512
```

## Output artifacts (where to look)

- Metrics are logged to wandb (if enabled) and also written under `simulator_output/<TIMESTAMP>/`.
- Vidur exports a Chrome trace for each run; open it via `chrome://tracing/` (or `edge://tracing/`) and load the trace file from `simulator_output/`.

## Common gotchas

- Python version: Vidur’s README documents Python 3.10 for its `venv` setup; if you hit dependency issues, use a Python 3.10 environment for Vidur rather than the project’s default Python.
- Profiling coverage: if you choose a new `model_name` or `device` without profiling data, you’ll need to generate profiling CSVs first (compute + optionally network).
- Wandb noise: disable wandb with `export WANDB_MODE=disabled` if you don’t want remote logging.

## Where to read next (sources)

- Vidur repo and run commands: https://github.com/microsoft/vidur
- Vidur paper (MLSys’24): https://arxiv.org/abs/2405.05465
- Metric definitions: https://github.com/microsoft/vidur/blob/main/docs/metrics.md
- Profiling guide (adding models/SKUs): https://github.com/microsoft/vidur/blob/main/docs/profiling.md
