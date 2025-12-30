# How to compare Vidur vs real timing for Qwen3 on A100

## HEADER
- **Purpose**: Step-by-step checklist to (1) simulate Qwen3 inference on an A100 using Vidur and (2) compare Vidur’s timing metrics against real A100 timing measurements using the same request-length + arrival process.
- **Status**: Draft (needs first end-to-end run with Qwen3 profiling data).
- **Date**: 2025-12-30
- **Dependencies**: `extern/tracked/vidur` submodule; an A100 machine for profiling + ground-truth timing; local model reference at `models/qwen3-0.6b/source-data`; optional `wandb` (can be disabled).
- **Target**: Contributors implementing reproducible simulator-vs-real comparisons.

## What Vidur needs (high level)

Vidur is a system-level simulator that predicts end-to-end latency/throughput using: (a) model + device configs, (b) profiled operator latency data (compute + optionally network), and (c) a workload generator (request lengths + arrival process + scheduler knobs). For “new model on a given SKU”, you typically do a one-time GPU profiling phase; then simulations can run CPU-only.

Sources:
- Vidur repo + CLI usage: https://github.com/microsoft/vidur
- Vidur metrics glossary (TTFT/TPOT/etc): https://github.com/microsoft/vidur/blob/main/docs/metrics.md
- Vidur profiling guide: https://github.com/microsoft/vidur/blob/main/docs/profiling.md
- Vidur paper (fidelity claims + methodology): https://arxiv.org/abs/2405.05465

## Step 0: Decide what you will compare

Pick one “deployment slice” and keep it fixed across real + sim:
- Hardware: `A100 80GB` (Vidur: `--replica_config_device a100`; real: confirm via `nvidia-smi`).
- Parallelism/topology: start with `TP=1`, `PP=1`, `num_replicas=1` (network profiling is irrelevant for TP1/PP1).
- Workload definition: request-length distribution (prefill tokens + decode tokens) + arrival process (e.g. Poisson at fixed QPS).
- Metric mapping: compare Vidur’s `prefill_time_e2e_*` (TTFT) and `decode_token_execution_plus_preemption_times` (inter-token delay / TPOT proxy) with your measured TTFT and per-token decode latency distributions.

## Step 1: Create a request-length trace CSV (shared by real + sim)

Vidur’s trace length generator expects a CSV with columns:
- `num_prefill_tokens`
- `num_decode_tokens`

Minimal example (generate from your real prompts):

```python
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

model_dir = Path("models/qwen3-0.6b/source-data").resolve()
tok = AutoTokenizer.from_pretrained(model_dir)

prompts = ["Hello", "Explain GPU simulation in 2 sentences.", "Write a haiku about schedulers."]
decode_tokens = 128  # or per-request values if you have them

rows = []
for p in prompts:
    prefill = len(tok(p).input_ids)
    rows.append({"num_prefill_tokens": prefill, "num_decode_tokens": decode_tokens})

pd.DataFrame(rows).to_csv("tmp/qwen3_trace_lengths.csv", index=False)
```

Notes:
- Keep tokenization consistent with what you use in the real benchmark (same tokenizer + same prompt text).
- If you measure real “stop early” behavior (EOS), reflect that in `num_decode_tokens` rather than always using `max_new_tokens`.

## Step 2: Measure “real” A100 timing for the same workload

You need per-request TTFT and per-token decode latencies under the same arrival process as the simulation (e.g. Poisson at QPS X). The most comparable approach is to benchmark a real serving stack (vLLM / Sarathi) with concurrency + batching enabled; a single-process `transformers.generate()` loop is useful for smoke tests but does not match system behavior under load.

Practical options:
- vLLM: run an OpenAI-compatible server and use its benchmark scripts/client to collect TTFT/TBT (time-between-tokens). Source: https://github.com/vllm-project/vllm
- Sarathi-Serve (used in Vidur’s profiling workflow): follow Vidur’s profiling doc which references the `sarathi-serve` repo/branch and use the same stack for ground truth. Source: https://github.com/microsoft/sarathi-serve (Vidur doc references a `vidur` branch)

If you do a quick local-only baseline with `transformers`, keep it explicit that it is “single-process, local generate” and likely not comparable to Vidur’s scheduler-level metrics.

## Step 3: Make Vidur recognize Qwen3-0.6B as a model_name

Vidur chooses model hyperparameters from Python dataclasses under `extern/tracked/vidur/vidur/config/model_config.py`, via `BaseModelConfig.create_from_name(model_name)`.

For Qwen3-0.6B, you need an entry with `get_name()` matching what you pass on the CLI (recommended: `Qwen/Qwen3-0.6B`) and parameters taken from the model’s `config.json` (this repo’s local copy is at `models/qwen3-0.6b/source-data/config.json`):
- `num_layers=28`, `num_q_heads=16`, `num_kv_heads=8`
- `embedding_dim=1024`, `mlp_hidden_dim=3072`
- `max_position_embeddings=40960`, `vocab_size=151936`, `rope_theta=1000000`

Important: Vidur’s profiling uses FP16 by default; for apples-to-apples comparisons, either benchmark real runs in FP16 or extend Vidur’s profiling/config to match BF16.

## Step 4: Generate Vidur profiling data for Qwen3 on A100

Vidur simulations require compute profiling CSVs under `extern/tracked/vidur/data/profiling/compute/a100/<model_name>/` (and network profiling if you use TP>1 or PP>1).

Follow Vidur’s profiling guide (source: https://github.com/microsoft/vidur/blob/main/docs/profiling.md). At a high level:
1) Set up `sarathi-serve` (Vidur branch) on an A100 machine.
2) Install Vidur into the same environment (`python -m pip install -e .` from `extern/tracked/vidur/`).
3) Run profiling scripts (MLP + attention) for your `model_name`.
4) Copy the resulting `mlp.csv` and `attention.csv` into Vidur’s `data/profiling/compute/a100/<model_name>/`.

Example commands (run inside `extern/tracked/vidur/` with the correct env activated):

```bash
python vidur/profiling/mlp/main.py --models Qwen/Qwen3-0.6B --num_gpus 1
python vidur/profiling/attention/main.py --models Qwen/Qwen3-0.6B --num_gpus 1
```

## Step 5: Run a Vidur simulation for Qwen3 on A100

Run from `extern/tracked/vidur/` (use Vidur’s recommended Python 3.10 env). Start by disabling wandb noise:

```bash
export WANDB_MODE=disabled
```

Then run a trace-driven workload with Poisson arrivals (edit paths/QPS/tokens to match your real benchmark):

```bash
python -m vidur.main \
  --replica_config_device a100 \
  --replica_config_model_name Qwen/Qwen3-0.6B \
  --cluster_config_num_replicas 1 \
  --replica_config_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests 512 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_trace_file /abs/path/to/tmp/qwen3_trace_lengths.csv \
  --trace_request_length_generator_config_max_tokens 40960 \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 1.0 \
  --replica_scheduler_config_type sarathi \
  --sarathi_scheduler_config_batch_size_cap 512 \
  --sarathi_scheduler_config_chunk_size 512
```

Outputs are written under `extern/tracked/vidur/simulator_output/<TIMESTAMP>/` and include CSVs and plots under `plots/` (see Vidur README: https://github.com/microsoft/vidur).

## Step 6: Compare results (recommended workflow)

1) Extract Vidur distributions from `extern/tracked/vidur/simulator_output/<TIMESTAMP>/plots/` and/or `request_metrics.csv`:
   - TTFT proxy: `prefill_time_e2e_cdf` (aka time-to-first-token from arrival)
   - Per-token decode latency proxy: `decode_token_execution_plus_preemption_times`
2) Extract real distributions from your benchmark logs (same definitions: arrival-to-first-token and time-between-tokens under load).
3) Compare percentiles (P50/P90/P99) and check shape differences; if they diverge, validate these alignment points first:
   - Same tokenizer + same `num_prefill_tokens`/`num_decode_tokens` trace
   - Same precision (FP16 vs BF16), same attention implementation assumptions
   - Same scheduler/batching knobs (batch size cap, chunk size, preemption behavior)
   - Same arrival process (QPS + distribution) and warmup (exclude cold-start effects)

## Known gaps / gotchas

- Vidur’s docs mention YAML model configs, but this repo’s Vidur version uses Python dataclasses in `vidur/config/model_config.py` for `model_name` resolution; plan your “add Qwen3 model config” work accordingly.
- Avoid editing `extern/tracked/vidur` directly unless you’re ready to carry a fork or a patch; consider a small wrapper entrypoint that registers a Qwen3 model config before invoking Vidur if you need to keep the submodule clean.
