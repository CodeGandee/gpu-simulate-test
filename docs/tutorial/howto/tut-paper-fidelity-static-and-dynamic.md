# How to produce paper-fidelity static and dynamic reports with `paper-fidelity`

## Question
How do I produce the paper-fidelity **static** and **dynamic** reports (Vidur sim vs Sarathi-Serve real replay) using this repo’s `paper-fidelity` CLI?

## Prerequisites

- **Environment:** `pixi` installed and `pixi install` has been run.
- **GPU:** A working CUDA setup (`pixi run python -c "import torch; print(torch.cuda.is_available())"` prints `True`).
- **GPU pinning:** Repo-local `.env` sets `GSIM_CUDA_VISIBLE_DEVICES` (see `context/instructions/prep-dev-env.md`).
- **Submodules:** `git submodule update --init --recursive` has been run (needed for Vidur + Sarathi).
- **Model assets:** The selected scenario’s `scenario.model.model_ref` exists (e.g., via `bash models/bootstrap.sh` or per-model bootstraps like `bash models/llama2-70b-hf/bootstrap.sh`).
- **Sarathi import works:** `pixi run python -c "import sarathi; print(sarathi.__version__ if hasattr(sarathi, '__version__') else 'ok')"` succeeds.
- **Host-matched profiling:** You will run `paper-fidelity profile` on this machine before running `paper-fidelity repro` (this is required to reproduce sim-vs-real % error meaningfully).

## Implementation Idea

**Approach:**
1. Pick a **scenario** (e.g., `llama2_7b_arxiv`) and (optionally) a **scale** (`small|medium|full`).
2. Generate a **host-matched Vidur profiling root** (microbenchmark bundle) for the scenario.
3. Run `paper-fidelity repro` once for **static** and once for **dynamic**, pointing Vidur at the host profiling root.
4. Read the generated report(s) under `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/…/summary.md`.

Key behaviors:
- **Static**: all requests arrive at time `0` (no capacity search).
- **Dynamic**: generates a **timed trace** (`trace.csv` with a non-zero `arrived_at` column) using a Poisson arrival process, and (by default for this scenario) runs a **capacity search** to choose an operating QPS (85% of capacity).
 - **Important:** To reproduce the *sim-vs-real percent error* in a comparable way, Vidur must be driven by a profiling bundle produced on the **same host + GPU stack** as the real replay. Using the paper-provided profiling bundle is useful for sanity checks, but it can change the meaning of the % error.

## Step-by-Step with Code

### Step 1: Confirm environment and assets

This makes sure you’re running inside the Pixi env, your submodules are present, and the model reference exists (Sarathi needs it to replay “real” timing).

```bash
git submodule update --init --recursive
pixi install

# Optional but recommended: ensure GSIM_CUDA_VISIBLE_DEVICES is set (repo-local).
test -f .env && sed -n '1,50p' .env || true

pixi run python -c "import torch; print('cuda_available=', torch.cuda.is_available())"

# Ensure model reference exists (symlink pattern; weights are machine-local).
bash models/llama2-7b-hf/bootstrap.sh
ls -la models/llama2-7b-hf/source-data
```

### Step 2: Generate a host-matched Vidur profiling root (required)

By default, the scenario points Vidur at the paper-provided profiling bundle under `extern/tracked/vidur`.
For sim-vs-real **% error reproduction**, you want a **host-matched** profiling root (microbenchmarks collected on this machine/GPU/runtime). Treat this as required.

```bash
# Creates a profiling root under `tmp/paper_fidelity/profiling_roots/<scenario>/<timestamp>/`.
pixi run paper-fidelity profile --scenario llama2_7b_arxiv

# Recommended: include CPU overhead microbenchmarks (required for the paper-model sweep workflow).
pixi run paper-fidelity profile --scenario llama2_7b_arxiv --include-cpu-overhead
```

Capture the printed directory path; you’ll pass it back as `scenario.vidur.profiling_root=...` in Step 3/4.

### Step 3: Produce the static report

This runs: trace generation → Vidur simulation → Sarathi replay → score → report.

```bash
pixi run paper-fidelity repro \
  --scenario llama2_7b_arxiv \
  --workload static \
  --scale small \
  "scenario.vidur.profiling_root=tmp/paper_fidelity/profiling_roots/llama2_7b_arxiv/<timestamp-dir>"
```

Expected output:
- Report directory printed on the last line (for example):
  - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/llama2_7b_arxiv/`
- Files inside the report directory:
  - `summary.md`, `run_meta.json`, `scores.json`, `figs/*.svg`, `inputs/*.csv`

### Step 4: Produce the dynamic report (small scale)

Dynamic mode supports `--scale`. For `small`, the run uses the first 50 requests after max-token filtering.

```bash
pixi run paper-fidelity repro \
  --scenario llama2_7b_arxiv \
  --workload dynamic \
  --scale small \
  "scenario.vidur.profiling_root=tmp/paper_fidelity/profiling_roots/llama2_7b_arxiv/<timestamp-dir>"
```

Expected output:
- Report directory printed on the last line (for example):
  - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/llama2_7b_arxiv_dynamic_small/`

### Step 4a: Paper models (InternLM-20B, LLaMA2-70B, Qwen-72B)

Scenarios added by this repo for the Vidur paper models:

- `internlm_20b_arxiv`
- `llama2_70b_arxiv`
- `qwen_72b_arxiv`

Static repro (small scale) for each:

```bash
# Run `paper-fidelity profile --scenario <scenario> --include-cpu-overhead` first and
# set PROFILING_ROOT to the printed path for that scenario.
PROFILING_ROOT="/abs/path/to/tmp/paper_fidelity/profiling_roots/<scenario>/<timestamp-dir>"

pixi run paper-fidelity repro --scenario internlm_20b_arxiv --workload static --scale small \
  "scenario.vidur.profiling_root=${PROFILING_ROOT}"

pixi run paper-fidelity repro --scenario llama2_70b_arxiv --workload static --scale small \
  "scenario.vidur.profiling_root=${PROFILING_ROOT}"

pixi run paper-fidelity repro --scenario qwen_72b_arxiv --workload static --scale small \
  "scenario.vidur.profiling_root=${PROFILING_ROOT}"
```

Dynamic repro (small scale) for each:

```bash
pixi run paper-fidelity repro --scenario internlm_20b_arxiv --workload dynamic --scale small \
  "scenario.vidur.profiling_root=${PROFILING_ROOT}"

pixi run paper-fidelity repro --scenario llama2_70b_arxiv --workload dynamic --scale small \
  "scenario.vidur.profiling_root=${PROFILING_ROOT}"

pixi run paper-fidelity repro --scenario qwen_72b_arxiv --workload dynamic --scale small \
  "scenario.vidur.profiling_root=${PROFILING_ROOT}"
```

### Step 4b: How the timed trace is generated (dynamic) and where to find it

**How it’s generated**

- The base trace comes from the scenario’s token-length source (`configs/paper_fidelity/scenario/llama2_7b_arxiv.yaml`), producing rows with:
  - `num_prefill_tokens`, `num_decode_tokens`, `request_id`, and `arrived_at=0`.
- For `--scale small`, the run subsets to the first 50 rows (`configs/paper_fidelity/scale/small.yaml`).
- For **dynamic**, the run assigns **arrival times** using `add_poisson_arrivals()`:
  - `arrived_at` is built from cumulative sums of exponential inter-arrival samples (mean `1 / qps`) with a fixed seed.
  - Source: `src/gpu_simulate_test/paper_fidelity/traces.py` (`add_poisson_arrivals`).
- In `paper-fidelity repro --workload dynamic`, the run first performs a **capacity search** and then generates the final timed trace at `qps_85` (85% of discovered capacity).
  - Capacity search: `src/gpu_simulate_test/paper_fidelity/capacity.py` (`discover_capacity`).

**Where it’s written**

- Canonical (mutable) trace outputs (overwritten on reruns):
  - `tmp/paper_fidelity/traces/<scenario.name>/trace.csv`
  - `tmp/paper_fidelity/traces/<scenario.name>/trace_meta.json`
- Capacity artifacts for dynamic runs:
  - `tmp/paper_fidelity/runs/<scenario.name>/capacity/capacity.json`
  - `tmp/paper_fidelity/runs/<scenario.name>/capacity/qps_*/` (per-probe Sarathi outputs)
- Report snapshot inputs (stable, per-report):
  - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<report_scenario>/inputs/trace.csv`
  - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<report_scenario>/inputs/trace_meta.json`
  - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<report_scenario>/inputs/capacity.json` (dynamic only)

**Example: timed `trace.csv`**

```csv
arrived_at,num_prefill_tokens,num_decode_tokens,request_id
0.0,3772,54,0
2.828480710548229,2015,156,1
5.576939129165233,3858,133,2
8.382540305487886,2509,79,3
```

**Example: `trace_meta.json`**

```json
{
  "artifacts": { "trace_csv": "/abs/path/to/tmp/paper_fidelity/traces/<scenario>/trace.csv" },
  "generated_at": "2026-01-12T15:24:17.365438+00:00",
  "qps": 0.85,
  "scale": "small",
  "scenario_name": "llama2_7b_arxiv",
  "schema_version": "v1",
  "seed": 42,
  "trace_source": {
    "kind": "vidur_processed_lengths_csv",
    "max_tokens": 4096,
    "num_requests": null,
    "path": "/abs/path/to/extern/tracked/vidur/data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv",
    "seed": 42
  },
  "trace_subset": { "begin": 0, "end": 50, "indices": null, "kind": "range" },
  "workload_mode": "dynamic"
}
```

**Example: `capacity.json`**

```json
{
  "capacity_qps": 1.0,
  "criterion": { "metric": "request_scheduling_delay", "quantile": 0.99, "threshold_s": 5.0 },
  "qps_85": 0.85
}
```

### Step 5: (Optional) Enable CPU overhead modeling (host profiling root required)

```bash
PROFILING_ROOT="tmp/paper_fidelity/profiling_roots/llama2_7b_arxiv/<timestamp-dir>"

pixi run paper-fidelity repro \
  --scenario llama2_7b_arxiv \
  --workload dynamic \
  --scale medium \
  "scenario.vidur.profiling_root=${PROFILING_ROOT}" \
  "scenario.vidur.skip_cpu_overhead_modeling=false" \
  "scenario.vidur.cpu_overhead.validation=strict"
```

Note: when you re-run on the same day with the same `scenario.name`, the report directory path can collide.
If you want runs side-by-side, override `scenario.name=...` (treat it as an artifact namespace).

### Complete Runnable Script

```bash
#!/usr/bin/env bash
set -euo pipefail

git submodule update --init --recursive
pixi install

# Ensure model reference exists for Sarathi replay.
bash models/llama2-7b-hf/bootstrap.sh >/dev/null

profiling_root="$(pixi run paper-fidelity profile --scenario llama2_7b_arxiv | tail -n 1)"

# Static report (default scale=full), using host-matched profiling.
pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload static \
  "scenario.vidur.profiling_root=${profiling_root}"

# Dynamic report (medium scale), using host-matched profiling.
pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic --scale medium \
  "scenario.vidur.profiling_root=${profiling_root}"
```

### [Optional] Alternative Interface (batch runner script)

To run **both workloads** across `small|medium|full` scales (and write a manifest), use:

```bash
bash scripts/run_pf_llama2_7b_sim_vs_real.sh
```

## Input and Output

### Input

- `--scenario` (string): scenario key under `configs/paper_fidelity/scenario/` (e.g., `llama2_7b_arxiv`).
- `--workload` (`static|dynamic`): workload mode.
- `--scale` (`small|medium|full`, optional): applies trace subsetting; also affects report naming for dynamic runs.
- Hydra overrides (optional): e.g., `scenario.vidur.profiling_root=...`, `scenario.name=...`.

### Output

- Report directory:
  - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<scenario_name_or_tag>/`
- Core files:
  - `summary.md`: human-readable report (tables + embedded SVG links)
  - `run_meta.json`: environment + params provenance
  - `scores.json`: scored percent-error metrics and verdicts
  - `inputs/`: snapshots of `sim_request_metrics.csv`, `real_request_metrics.csv` (and trace/capacity inputs when present)
  - `figs/`: SVG figures (ECDF + percentiles)
  - `tmp/paper_fidelity/traces/<scenario.name>/trace.csv`: canonical trace used by Vidur + Sarathi (note: overwritten on reruns; use the report snapshot under `inputs/` for stability)

## References

### Relevant Source Code

- `src/gpu_simulate_test/cli/paper_fidelity.py`: `paper-fidelity` CLI and end-to-end `repro` orchestration.
- `src/gpu_simulate_test/paper_fidelity/report.py`: report writer (`summary.md`, `scores.json`, figures).
- `src/gpu_simulate_test/paper_fidelity/capacity.py`: capacity search used by dynamic workloads.
- `src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py`: Sarathi replay runner (real metrics).
- `src/gpu_simulate_test/vidur_ext/sim_runner.py`: Vidur paper-fidelity simulator wrapper (sim metrics).

### Online Resources

- Vidur simulator: `extern/tracked/vidur/`
- Sarathi-Serve: `extern/tracked/sarathi-serve/`
