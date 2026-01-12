# How to use Vidur for LLaMA2-7B (end-to-end, sim vs real, with a worked example)

This tutorial is written for a Python machine learning (ML) developer who has **no Vidur experience** but understands large language model (LLM) inference concepts (prefill vs decode, batching, key/value (KV) cache, tensor parallelism (TP), pipeline parallelism (PP)).

It uses this repo’s canonical worked example report as the “target output”:

- `results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv/`
- Companion report guide: `results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv/intro-to-the-report.md`

The goal is not just to run the commands in this repo, but to understand the **portable reasons** behind each step so you can replicate the approach in your own project.

---

## 0) A portable mental model (what you’re doing, conceptually)

### What Vidur does (concept)

Vidur is an event-driven simulator for LLM inference that predicts latency/throughput by combining:

- A scheduling model (how requests are batched, chunked, preempted, etc.).
- A runtime model (how long prefill/decode “work” takes), typically derived from profiling data.

It does **not** generate model outputs; it predicts timing using models.

### What a “profiling root/bundle” is (concept)

To be credible, Vidur needs calibration data that matches:

- model (e.g., `meta-llama/Llama-2-7b-hf`)
- hardware (e.g., NVIDIA A100-SXM4-80GB)
- topology (e.g., NVLink communication assumptions)
- parallelism (TP/PP)
- kernel stack (the real engine and attention backend can matter)

In this repo, that calibration data is stored as a Vidur-compatible directory tree under:

- `tmp/paper_fidelity/profiling_roots/<scenario>/<run_id>/data/profiling/...`

### What “sim vs real” means (concept)

“Sim vs real” is only meaningful if both sides see the *same workload* and measure a comparable boundary:

- same per-request lengths (prefill tokens, decode tokens)
- same arrivals (static vs Poisson)
- aligned scheduler knobs (chunk size, max inflight sequences, etc.)
- aligned stopping behavior (avoid early end-of-sequence (EOS) stopping unless both sides model it)

This repo’s “real” side uses Sarathi-Serve replay + metrics (and writes a `request_metrics.csv`).

---

## 1) Prerequisites (portable first, then this repo’s implementation)

### Portable prerequisites (concept)

You need:

- A working Compute Unified Device Architecture (CUDA) driver/runtime and a matching PyTorch build.
- A reliable Graphics Processing Unit (GPU) selection strategy for multi-GPU/multi-instance machines (worker processes must see a “healthy” subset).
- Local model assets for the “real” runner (weights/tokenizer), or a clearly documented alternative that is identical between profiling and real replay.

### This repo’s implementation (example)

Environment and dependencies:

```bash
git submodule update --init --recursive
pixi install
```

GPU pinning via environment variables (env vars):

- Put a repo-local `.env` in the repo root (not committed):

```bash
export GSIM_CUDA_VISIBLE_DEVICES=0
```

- Sanity check inside Pixi:

```bash
pixi run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected output pattern:

```text
2.9.1+cu128
True 1
```

Model assets for the real runner:

- `configs/paper_fidelity/scenario/llama2_7b_arxiv.yaml` points `scenario.model.model_ref` at:
  - `models/llama2-7b-hf/source-data`

If this is missing, follow `context/instructions/prep-dev-env.md`.

---

## 2) Worked example: reproduce the LLaMA2-7B Arxiv report

This section reproduces the *kind* of report found in:

- `results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv/`

The date/run ids will differ when you run it again.

### Step 1: Generate a host profiling root (includes CPU overhead)

#### Why this exists (portable concept)

If your profiling bundle does not match your host/runtime, Vidur can look “too fast” because it is missing real-world overheads and kernel behavior. CPU-side overhead is a common culprit.

#### This repo’s implementation (command)

```bash
pixi run paper-fidelity profile --scenario llama2_7b_arxiv --include-cpu-overhead
```

Example output (last line is the profiling root path):

```text
/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/profiling_roots/llama2_7b_arxiv/2026-01-12_04-06-24-491602
```

### Step 2: Verify the CPU overhead calibration is present and non-placeholder

#### Why this exists (portable concept)

If you “enable CPU overhead modeling” but your `cpu_overheads.csv` is missing/empty/placeholder, you can get misleading underprediction that looks like a simulator fidelity gap.

#### This repo’s implementation (commands)

```bash
profiling_root="tmp/paper_fidelity/profiling_roots/llama2_7b_arxiv/<your-run-id>"
cpu_csv="$profiling_root/data/profiling/cpu_overhead/a100_pairwise_nvlink/meta-llama/Llama-2-7b-hf/cpu_overheads.csv"
ls -la "$cpu_csv"
head -n 5 "$cpu_csv"
```

Example output (comma-separated values (CSV) header + a few rows):

```text
schedule_mean,sampler_e2e_mean,prepare_inputs_e2e_mean,model_execution_e2e_mean,...
0.0426207,0.3564672,0.1117332,15.0993323,...,meta-llama/Llama-2-7b-hf,8,1,2.33972
...
```

### Step 3: Run sim vs real reproduction (static, small) with CPU overhead modeling enabled

#### Why this exists (portable concept)

You want to compare simulation vs a real engine run for the same workload. “Static small” is a bounded run that still exercises the full pipeline.

#### This repo’s implementation (command)

```bash
pixi run paper-fidelity repro \
  --scenario llama2_7b_arxiv \
  --workload static \
  --scale small \
  "scenario.vidur.profiling_root=$profiling_root" \
  "scenario.vidur.skip_cpu_overhead_modeling=false"
```

Example output (printed report directory):

```text
/data1/huangzhe/code/gpu-simulate-test/results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv
```

### Step 4: Read the report and confirm it is “credible”

#### Why this exists (portable concept)

Before interpreting percent errors, confirm you didn’t accidentally run an incomparable experiment (wrong profiling root, CPU overhead missing, token semantics mismatch).

#### This repo’s implementation (quick checks)

```bash
sed -n '1,120p' results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv/summary.md
```

Things to look for in the `## Profiling` section:

- `mode: host` (host-matched profiling root)
- `cpu_overhead: status: ok` (CPU overhead modeling has real inputs)

Example score lines (percent error thresholds are in `run_meta.json`):

```text
| request_execution_plus_preemption_time_normalized | p50 | ... | ... | 3.38% | pass |
| request_e2e_time_normalized                      | p50 | ... | ... | 0.91% | pass |
```

---

## 3) Interpreting the metrics (what “normalized” means)

This repo’s paper-fidelity report focuses on per-token normalized metrics to make different output lengths comparable.

For Sarathi-Serve (source of truth for the “real” metric semantics), key normalizations are:

- `request_e2e_time_normalized = request_e2e_time / num_output_tokens`
- `request_execution_plus_preemption_time_normalized = (execution_time + preemption_time) / num_output_tokens`
- `prefill_time_execution_plus_preemption_normalized = prefill_execution_plus_preemption_time / num_prompt_tokens`
- `decode_time_execution_plus_preemption_normalized = decode_execution_plus_preemption_time / num_output_tokens`

Percent error is computed as:

```text
percent_error = abs(sim - real) / abs(real)
```

---

## 4) Figures (empirical cumulative distribution function plots)

The worked example includes empirical cumulative distribution function (ECDF) and percentile plots.
Copies are placed next to this tutorial for convenience:

- `context/summaries/vidur-kb/tutorials/figures/request_execution_plus_preemption_time_normalized_ecdf.svg`
- `context/summaries/vidur-kb/tutorials/figures/request_execution_plus_preemption_time_normalized_percentiles.svg`
- `context/summaries/vidur-kb/tutorials/figures/request_e2e_time_normalized_ecdf.svg`
- `context/summaries/vidur-kb/tutorials/figures/request_e2e_time_normalized_percentiles.svg`

These plots are most useful for:

- spotting distributional skew (not just matching percentiles)
- checking whether mismatch is concentrated in the tail (p95) or uniform

---

## 5) Caveats and easy-to-make mistakes (and how to avoid them)

This section is intentionally “sharp”: these are common ways to produce misleading results.

### Mistake A: Enabling CPU overhead modeling with dummy/missing CPU overhead inputs

- How to recognize:
  - `summary.md` shows CPU overhead `status: missing` / `placeholder` / `error`.
  - Simulation is systematically faster than real across metrics.
- How to fix:
  - Rerun profiling with `--include-cpu-overhead` and confirm `cpu_overheads.csv` exists and is non-empty.
- How to prevent:
  - Keep validation `strict` unless you are explicitly debugging.

### Mistake B: Worker processes see different GPUs than you think (Ray/Sarathi GPU visibility)

- How to recognize:
  - Logs mention `RuntimeError: No CUDA GPUs are available` inside a worker process.
- How to fix:
  - Pin a healthy subset with `GSIM_CUDA_VISIBLE_DEVICES` and rerun.
  - Inspect Ray logs under `/tmp/ray/session_latest/logs/` for the first failing worker.
- How to prevent:
  - Always pin GPUs (don’t rely on “whatever is visible on the host”).

### Mistake C: Relying on defaults for parity-critical scheduler knobs

- How to recognize:
  - Headline metric looks “okay”, but prefill vs decode split is badly wrong (prefill underestimated, decode overestimated, or vice versa).
- How to fix:
  - Explicitly set and record chunk size and batch caps on both sim and real sides.
  - In this repo, those are in `configs/paper_fidelity/scenario/llama2_7b_arxiv.yaml` under `scenario.real.scheduler.*` and `scenario.vidur.scheduler.*`.
- How to prevent:
  - Treat defaults as unsafe for fidelity work; always set the knobs in config and confirm they appear in `run_meta.json`.

### Mistake D: Token-count mismatch from early stopping (EOS)

- How to recognize:
  - Sim and real produce different `request_num_decode_tokens` for the same request ids; comparisons become invalid.
- How to fix:
  - Disable early EOS stopping in the real runner (this repo uses `ignore_eos: true`).
- How to prevent:
  - Treat “token-count match” as a hard validation step, not a best-effort check.

### Mistake E: Using a non-host-matched profiling root for host-fidelity claims

- How to recognize:
  - Profiling root path points to a “paper bundle” or a different host/topology than your run.
  - Results may look “reasonable” but drift with hardware changes in ways that aren’t explainable.
- How to fix:
  - Regenerate profiling root on the target host for the target model/topology.
- How to prevent:
  - Record profiling-root provenance (commit, host, GPU) and treat it as part of the experiment definition.

---

## 6) Porting the workflow to your own project (what to copy, conceptually)

If you are not using this repo, the minimal portable recipe is:

1) Define a canonical workload format: a trace with `arrived_at`, `num_prefill_tokens`, `num_decode_tokens`.
2) Ensure real replay uses the *same* token semantics (avoid early EOS unless modeled).
3) Build or obtain a profiling bundle that matches model/hardware/topology and includes CPU/runtime overhead if you care about wall-clock fidelity.
4) Run Vidur simulation against that trace and profiling bundle.
5) Run real replay, export per-request metrics with consistent definitions.
6) Compare distributions and percentiles, and record full provenance.

In this repo, these concepts map to:

- Trace: `tmp/paper_fidelity/traces/llama2_7b_arxiv/trace.csv`
- Profiling root: `tmp/paper_fidelity/profiling_roots/llama2_7b_arxiv/...`
- Real replay: Sarathi backend via `paper-fidelity repro`
- Report: `results/reports/<date>/paper_fidelity/llama2_7b_arxiv/`
