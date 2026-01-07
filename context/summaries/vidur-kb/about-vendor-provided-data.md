# Vidur “vendor-provided” data in `extern/tracked/vidur/data/`

This repo vendors Vidur as a git submodule under `extern/tracked/vidur/`. Vidur ships a small set of **workload traces** and **profiling bundles** under `extern/tracked/vidur/data/` that are sufficient to run the simulator without access to GPUs (after profiling is produced).

At a high level:

- `extern/tracked/vidur/data/processed_traces/`: CSVs describing **request token lengths** (and sometimes **arrival times**) for workloads.
- `extern/tracked/vidur/data/profiling/`: CSVs describing **per-op execution times** (compute) and **collective/network times** (communication), used by the simulator’s execution-time predictor.

## Directory layout

```text
extern/tracked/vidur/data/
  processed_traces/
    *.csv
  profiling/
    compute/
      <device>/                # a100, h100, a40, ...
        <org>/<model>/         # e.g. meta-llama/Llama-2-70b-hf/
          mlp.csv
          attention.csv
    network/
      <network_device>/        # e.g. a100_pairwise_nvlink, a100_dgx, ...
        all_reduce.csv
        send_recv.csv
    # cpu_overhead/            # path exists in Vidur defaults, but may not be shipped here
```

## Processed traces (`data/processed_traces/`)

Vidur uses “processed traces” as **token-length distributions** (and optionally arrival timestamps). These traces are already tokenized/filtered/trimmed to match the paper’s evaluation setup (e.g., 4K max context for LLaMA2-family experiments).

### Two ways traces are consumed

Vidur has two main trace ingestion modes (terminology based on Vidur config classes):

1. **Trace request *length* generator** (`TraceRequestLengthGenerator`)
   - Used with the “synthetic” request generator (`--request_generator_config_type synthetic`), where arrivals are produced by an interval generator (static/Poisson/etc).
   - Reads a CSV and samples `(num_prefill_tokens, num_decode_tokens)` pairs.
   - Required columns: `num_prefill_tokens`, `num_decode_tokens`.
   - Extra columns are allowed (they will be ignored).
   - Implementation: `extern/tracked/vidur/vidur/request_generator/trace_request_length_generator.py`.

2. **Trace *replay* request generator** (`TraceReplayRequestGenerator`)
   - Replays arrivals directly from the trace.
   - Required columns: `arrived_at`, `num_prefill_tokens`, `num_decode_tokens`.
   - `arrived_at` is interpreted as seconds from the beginning of the trace and can be rescaled via `time_scale_factor`.
   - Implementation: `extern/tracked/vidur/vidur/request_generator/trace_replay_request_generator.py`.

### Column definitions (trace CSVs)

Common columns you will see:

- `num_prefill_tokens`: number of prompt (prefill) tokens for the request.
- `num_decode_tokens`: number of output (decode) tokens for the request.
- `arrived_at` (optional): request arrival timestamp in seconds (used for replay).

Optional/derived columns that may exist (example: Arxiv trace):

- `num_total_tokens`: `num_prefill_tokens + num_decode_tokens`.
- `pd_ratio`: prompt-to-decode ratio (`num_prefill_tokens / num_decode_tokens`).

### Token truncation and scaling semantics

Both the length generator and the replay generator support:

- **Scaling factors** (`prefill_scale_factor`, `decode_scale_factor`) to multiply token counts.
- A **`max_tokens` cap** to ensure `num_prefill_tokens + num_decode_tokens <= max_tokens`:
  - For the *length* generator, Vidur deducts overflow **proportionally** from prefill/decode to fit `max_tokens`, and then clips both to be at least 1.
  - For the *replay* generator, Vidur deducts overflow from **prefill** tokens (it does **not** adjust `num_decode_tokens`). Ensure your trace never has `num_decode_tokens > max_tokens - 1`, otherwise the replay generator can make `num_prefill_tokens` non-positive.

See:
- `extern/tracked/vidur/vidur/request_generator/trace_request_length_generator.py`
- `extern/tracked/vidur/vidur/request_generator/trace_replay_request_generator.py`

### What traces are shipped in this repo snapshot

This repo’s Vidur submodule snapshot currently includes:

- `arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv`
  - Columns: `num_prefill_tokens`, `num_decode_tokens`, plus derived stats.
  - Suitable for **trace length** sampling (synthetic + static/Poisson arrivals).
- `splitwise_conv.csv`, `splitwise_code.csv`
  - Columns: `arrived_at`, `num_prefill_tokens`, `num_decode_tokens`.
  - Suitable for **trace replay** and also works for trace length sampling.

The Vidur paper references additional traces (e.g., Chat-1M, BWB-4K). Those may exist upstream but are not necessarily included in this submodule snapshot.

## Profiling bundles (`data/profiling/`)

Vidur’s simulator predicts per-request timing by training regressors on profiling CSVs. The key idea is to decompose per-token latency into:

- **Compute time** (MLP + attention) measured per model on a given GPU SKU.
- **Communication time** (collectives for tensor/pipeline parallelism) measured for a given node/network topology.

The default file templates (from Vidur config) are:

- Compute: `./data/profiling/compute/{DEVICE}/{MODEL}/mlp.csv`
- Compute: `./data/profiling/compute/{DEVICE}/{MODEL}/attention.csv`
- Network: `./data/profiling/network/{NETWORK_DEVICE}/all_reduce.csv`
- Network: `./data/profiling/network/{NETWORK_DEVICE}/send_recv.csv`
- CPU overhead (optional): `./data/profiling/cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv`

Placeholders are literal string replacements performed by Vidur:

- `{DEVICE}` ← `replica_config.device` (e.g., `a100`, `h100`, `a40`)
- `{MODEL}` ← model name (e.g., `meta-llama/Llama-2-70b-hf`)
- `{NETWORK_DEVICE}` ← `replica_config.network_device` (e.g., `a100_pairwise_nvlink`, `a100_dgx`)

Implementation: `extern/tracked/vidur/vidur/execution_time_predictor/sklearn_execution_time_predictor.py` (`_get_input_files()`).

### Compute profiling (`profiling/compute/<device>/<org>/<model>/`)

Files:

- `mlp.csv`: measurements for transformer “dense”/MLP-side operations (plus some per-layer scaffolding).
- `attention.csv`: measurements for attention-side operations for both prefill and decode shapes.

These CSVs contain many `time_stats.<op>.<stat>` columns (min/max/mean/median/std) and feature columns
describing the shape/context (e.g., number of heads, embedding dim, batch size, KV-cache size, whether
the row is decode vs prefill).

Vidur uses these rows to train regressors to predict op latency under the current simulation config.
The exact feature filtering and model fitting logic lives in:

- `extern/tracked/vidur/vidur/execution_time_predictor/sklearn_execution_time_predictor.py`

**Important distinction** (also noted in Vidur docs): compute profiling depends on **GPU SKU** (A100 vs H100),
and does *not* depend on the network topology (DGX vs pairwise NVLink), so it is keyed only by `<device>`.

### Network profiling (`profiling/network/<network_device>/`)

Files:

- `all_reduce.csv`: collective timing data used for **tensor parallel** communication.
- `send_recv.csv`: point-to-point timing data used for **pipeline parallel** stage transfers.

These CSVs contain `time_stats.<collective>.<stat>` columns and metadata like `num_workers`, `devices_per_node`,
and message `size`. Unlike compute profiling, network profiling depends strongly on node topology, so it is keyed
by `<network_device>` (e.g., DGX vs pairwise NVLink).

### CPU overhead profiling (often absent)

Vidur has optional CPU/runtime overhead modeling (scheduler overheads, sampling overheads, etc.), with a default
path under `data/profiling/cpu_overhead/...`.

This submodule snapshot does not necessarily ship `cpu_overhead` CSVs, and Vidur defaults to skipping CPU overhead
modeling (`skip_cpu_overhead_modeling=true` in the execution-time predictor config).

See:

- Default path template: `extern/tracked/vidur/vidur/config/config.py` (`cpu_overhead_input_file`)
- Behavior: `extern/tracked/vidur/vidur/execution_time_predictor/sklearn_execution_time_predictor.py`
- Vidur docs note: `extern/tracked/vidur/docs/profiling.md` (“CPU Overhead Profiling”)

## Practical mapping: from “paper config knobs” to data files

When you choose a simulation configuration, these are the key mappings to vendor-provided data:

- **Workload trace** → pick a `processed_traces/*.csv` file.
- **GPU SKU** (`a100`/`h100`/`a40`) → selects `profiling/compute/<device>/...` and also sets default `network_device`.
- **Model name** (HF id) → selects `profiling/compute/<device>/<org>/<model>/{mlp,attention}.csv`.
- **Node topology** (`network_device`) → selects `profiling/network/<network_device>/{all_reduce,send_recv}.csv`.

For examples of invoking Vidur with these artifacts, see `extern/tracked/vidur/README.md`.

## How the profiling data is obtained (paper + repo tooling)

Vidur’s profiling bundles are produced by running **microbenchmarks** on real hardware, then saving the measured
latencies to CSV.

### Methodology in the paper

In the paper’s design section, Vidur’s profiler:

- Classifies operators into **token-level**, **sequence-level (attention)**, and **communication** operators.
- Profiles only a **limited set** of input shapes and uses lightweight ML models (random forests) to interpolate.
- Profiles token-level ops using standard PyTorch kernels and measures runtime via CUPTI (via PyTorch’s profiling stack).
- Profiles attention separately for **prefill** and **decode**.
- Profiles communication collectives (e.g., **all-reduce**, **all-gather**, **send-recv**) in a **model-agnostic** way for each topology.

Relevant paper text:

- `extern/tracked/vidur/paper/tex/3-design.tex` (`\\subsection{\\profiler}` / `\\vheading{Profiling Communication Operators}`)
- `extern/tracked/vidur/paper/tex/5-eval.tex` (environment description: A100/H100 pairwise NVLink)

### Tooling in the repo (what you actually run)

Vidur ships profiling scripts under `extern/tracked/vidur/vidur/profiling/`:

- **Compute (MLP)**: `python extern/tracked/vidur/vidur/profiling/mlp/main.py ...`
  - Output: `profiling_outputs/mlp/<timestamp>/<model>/mlp.csv`
- **Compute (attention)**: `python extern/tracked/vidur/vidur/profiling/attention/main.py ...`
  - Output: `profiling_outputs/attention/<timestamp>/<model>/attention.csv`
- **Network (collectives)**: `python extern/tracked/vidur/vidur/profiling/collectives/main.py --collective all_reduce|send_recv ...`
  - Output: `profiling_outputs/collective/<timestamp>/all_reduce.csv` and/or `send_recv.csv`
- **CPU overhead (optional)**: `python extern/tracked/vidur/vidur/profiling/cpu_overhead/main.py ...`
  - Output: `profiling_outputs/cpu_overhead/<timestamp>/<model>/cpu_overhead.csv`
  - Note: the simulator’s default input template expects `cpu_overheads.csv`; you may need to rename the output or override the input path.

Once generated, the “vendor-provided” layout under `extern/tracked/vidur/data/profiling/` is just the **stable place**
to store these CSVs so simulations can run without GPUs.

## How the trace data is obtained (paper + repo assumptions)

### Methodology in the paper

In the evaluation setup, the paper says it “generates traces by using the request length characteristics” from:

- **Chat-1M** (multi-round conversations; each interaction round is a request)
- **Arxiv summarization** (large prompts, smaller outputs)
- **BWB** (translation-like; outputs often larger than prompts)

Then it **caps total request length to 4096 tokens** (LLaMA2 max context) and refers to the capped variants as the
“*-4K” traces.

Source: `extern/tracked/vidur/paper/tex/5-eval.tex` (`\\vheading{Workloads.}`), plus workload stats in
`extern/tracked/vidur/paper/tex/tables/bench_workloads.tex`.

Operationally, this means the “paper traces” are primarily:

- A distribution of `(num_prefill_tokens, num_decode_tokens)` pairs
- Combined with a *synthetic* arrival process (static/offline or Poisson/online), rather than true per-request timestamps

### What Vidur expects as input

Vidur’s simulator does not require the raw datasets. It only requires the **processed trace CSV** with the columns
described above (token lengths, and optionally arrival times).

In this vendored snapshot, `extern/tracked/vidur/data/processed_traces/` contains only a subset of the traces
referenced in the paper (e.g., the Arxiv length trace is present; Chat-1M and BWB traces may be missing).

## Onboarding checklist: new environment + new model

This is the practical “what do I run” checklist to reproduce Vidur-style vendor data for a new setup.

### 1) Define identifiers (device + network_device + model name)

- Pick `replica_config_device` (`a100`, `h100`, `a40`, …) and `replica_config_network_device`
  (`a100_pairwise_nvlink`, `a100_dgx`, …) such that:
  - There are matching SKU configs in:
    - `extern/tracked/vidur/vidur/config/device_sku_config.py`
    - `extern/tracked/vidur/vidur/config/node_sku_config.py`
  - There are matching folders under:
    - `extern/tracked/vidur/data/profiling/compute/<device>/...`
    - `extern/tracked/vidur/data/profiling/network/<network_device>/...`

If your GPU SKU or topology is new, you must add a new DeviceSKU/NodeSKU type and config (and create new profiling
folders) before simulation will work.

### 2) Add the model architecture config (required for profiling + simulation)

In this vendored Vidur snapshot, model configs are code-defined (not YAML). Add a new `BaseModelConfig` subclass in:

- `extern/tracked/vidur/vidur/config/model_config.py`

Make sure `get_name()` returns the exact HuggingFace model id you will pass as `--replica_config_model_name`
(and to the profiling scripts).

Note: `extern/tracked/vidur/docs/profiling.md` describes adding `data/model_configs/*.yml`, but that directory is not
present in this snapshot; the source of truth here is `vidur/config/model_config.py`.

### 3) Produce compute profiling (MLP + attention)

Run on a machine with the target GPU SKU (A100/H100/…).

MLP profiling:

```bash
python extern/tracked/vidur/vidur/profiling/mlp/main.py \
  --models <hf_model_id> \
  --num_gpus 1 \
  --num_tensor_parallel_workers 1 2 4 \
  --max_tokens 4096
```

Attention profiling:

```bash
python extern/tracked/vidur/vidur/profiling/attention/main.py \
  --models <hf_model_id> \
  --num_gpus 1 \
  --num_tensor_parallel_workers 1 2 4 \
  --max_model_len 4096 \
  --max_seq_len 4096 \
  --max_batch_size 128
```

Then copy outputs into Vidur’s expected layout (create folders as needed):

- `profiling_outputs/mlp/<timestamp>/<hf_model_id>/mlp.csv` → `extern/tracked/vidur/data/profiling/compute/<device>/<org>/<model>/mlp.csv`
- `profiling_outputs/attention/<timestamp>/<hf_model_id>/attention.csv` → `extern/tracked/vidur/data/profiling/compute/<device>/<org>/<model>/attention.csv`

### 4) Produce network profiling (all_reduce + send_recv)

Network profiling is model-agnostic but topology-specific. It uses Ray to coordinate the run.

If you only care about TP (no PP), you need only `all_reduce`. If you want PP, also profile `send_recv`.

Example (single node):

```bash
python extern/tracked/vidur/vidur/profiling/collectives/main.py \
  --num_workers_per_node_combinations 1 2 4 \
  --collective all_reduce

python extern/tracked/vidur/vidur/profiling/collectives/main.py \
  --num_workers_per_node_combinations 1 2 4 \
  --collective send_recv
```

For multi-node PP profiling, set up a Ray cluster first (see `extern/tracked/vidur/docs/profiling.md`), then run the same
commands.

Copy outputs:

- `profiling_outputs/collective/<timestamp>/all_reduce.csv` → `extern/tracked/vidur/data/profiling/network/<network_device>/all_reduce.csv`
- `profiling_outputs/collective/<timestamp>/send_recv.csv` → `extern/tracked/vidur/data/profiling/network/<network_device>/send_recv.csv`

### 5) (Optional) CPU overhead profiling

The paper notes it evaluates on an optimized vLLM fork that eliminates unnecessary CPU overheads, and Vidur defaults to
`skip_cpu_overhead_modeling=true`. If you need CPU overhead fidelity for a specific framework/runtime, use:

```bash
python extern/tracked/vidur/vidur/profiling/cpu_overhead/main.py --models <hf_model_id>
```

Then either rename `cpu_overhead.csv` → `cpu_overheads.csv`, or override the simulator config
(`--random_forrest_execution_time_predictor_config_cpu_overhead_input_file ...`) to point at the generated file.

### 6) Produce processed traces for the new model/workload

Vidur does not require full prompts/outputs; it only needs token counts. For best fidelity, tokenize using the **same
tokenizer as the model you plan to serve** (token counts differ across model families).

- For **synthetic workloads** (static or Poisson arrivals): write a CSV with:
  - `num_prefill_tokens`, `num_decode_tokens`
- For **replay workloads** (real arrivals): write a CSV with:
  - `arrived_at` (seconds since trace start), `num_prefill_tokens`, `num_decode_tokens`

If you are replicating the paper’s setup for LLaMA2-family workloads, cap to 4096 total tokens (or your model’s context
limit). If you will use `TraceReplayRequestGenerator`, ensure `num_decode_tokens <= max_tokens - 1`.

### 7) Sanity-check by running a simulation

From `extern/tracked/vidur/`, run a small simulation referencing your new data (paths shown here use the vendored layout):

```bash
python -m vidur.main \
  --replica_config_device <device> \
  --replica_config_network_device <network_device> \
  --replica_config_model_name <hf_model_id> \
  --request_generator_config_type synthetic \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/<your_trace>.csv
```

If Vidur errors on missing profiling files, it’s almost always a mismatch between:

- the `<device>/<model>` directory names, and `replica_config_device` / `replica_config_model_name`
- the `<network_device>` directory name, and `replica_config_network_device`
