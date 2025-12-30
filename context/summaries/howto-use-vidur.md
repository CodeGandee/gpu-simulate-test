# How to use Vidur (`vidur`) in this repo

## HEADER
- **Purpose**: Basic “it runs” usage notes for Vidur (Microsoft’s LLM inference system simulator) inside this repo’s Pixi environment.
- **Status**: Active
- **Date**: 2025-12-30
- **Dependencies**:
  - `pyproject.toml` (Pixi env; `vidur` installed editable from `extern/tracked/vidur`)
  - `extern/tracked/vidur/` (Vidur submodule source tree and default profiling data under `data/`)
- **Target**: Contributors running local simulations and iterating on simulator-vs-real comparisons.

## 1) Setup (Pixi + submodules)

1. Initialize submodules (if needed):
   - `git submodule update --init --recursive`
2. Install/update the Pixi environment:
   - `pixi install`
3. Sanity-check import (Vidur is installed editable):
   - `pixi run python -c "import vidur; print(vidur.__file__)"`

## 2) Quick smoke test (verified)

Vidur’s default profiling paths are relative to the current working directory (e.g. `./data/profiling/...`), so the most reliable smoke test is to run from the Vidur submodule root.

Command (verified to run successfully in this repo on 2025-12-30):

```bash
cd extern/tracked/vidur
WANDB_MODE=disabled pixi run python -m vidur.main \
  --synthetic_request_generator_config_num_requests 2 \
  --metrics_config_output_dir ../../../tmp/vidur_smoke \
  --metrics_config_cache_dir ../../../tmp/vidur_cache \
  --no-metrics_config_enable_chrome_trace \
  --no-metrics_config_store_plots
```

Expected result:

- Creates `tmp/vidur_smoke/<timestamp>/` with `config.json`, `request_metrics.csv`, and `plots/`.

## 3) Common knobs you’ll use

- Model + device:
  - `--replica_config_model_name <name>` (must match a Vidur model config `get_name()`)
  - `--replica_config_device a100|h100|...`
- Topology:
  - `--replica_config_tensor_parallel_size <TP>`
  - `--replica_config_num_pipeline_stages <PP>`
  - `--cluster_config_num_replicas <replicas>`
- Workload:
  - `--request_generator_config_type synthetic`
  - `--synthetic_request_generator_config_num_requests <N>`
- Output:
  - `--metrics_config_output_dir tmp/vidur_runs` (Vidur appends a timestamp subdir)

## 4) Trace-driven request lengths + arrivals

For “real vs sim” comparisons you typically want trace-driven request lengths and a controlled arrival process:

- Length trace CSV must have columns:
  - `num_prefill_tokens`
  - `num_decode_tokens`
- Use the trace length generator:
  - `--length_generator_config_type trace`
  - `--trace_request_length_generator_config_trace_file /abs/path/to/trace_lengths.csv`
- For arrivals, start with Poisson:
  - `--interval_generator_config_type poisson`
  - `--poisson_request_interval_generator_config_qps <qps>`

## 5) Gotchas

- **CWD matters**: if you run `pixi run python -m vidur.main` from repo root without overriding predictor input files, Vidur will look for profiling under `./data/profiling/...` (repo root), not under `extern/tracked/vidur/data/...`.
- **Wandb**: Vidur only initializes wandb if `metrics_config_wandb_project` and `metrics_config_wandb_group` are set, but `WANDB_MODE=disabled` is a good default for local runs.
- **Model-name resolution**: `--replica_config_model_name` must match a registered model config in Vidur (`vidur/config/model_config.py`). For new models (e.g. Qwen3), plan to add a config or register one via a wrapper entrypoint.
