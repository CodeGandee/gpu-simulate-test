# Intro to the report: `paper_fidelity/llama2_7b_arxiv` (2026-01-12)

This document explains how to reproduce and interpret the paper-fidelity report in:

- `results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv/`

The workflow compares **Vidur simulation** vs **Sarathi “real” execution** for the same scenario and trace, then scores the percent error at key percentiles.

---

## Prerequisites

### Repo + dependencies

- Submodules initialized:
  - `git submodule update --init --recursive`
- Pixi environment installed:
  - `pixi install`
- Run commands via Pixi (don’t use system Python):
  - `pixi run ...`

Example sanity check:

```bash
pixi run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Example output:

```text
2.9.1+cu128
True 1
```

### GPU pinning (required)

This repo enforces explicit pinning for GPU work via `GSIM_CUDA_VISIBLE_DEVICES` (loaded from a repo-local `.env` if present).

Example `.env` (repo root):

```bash
export GSIM_CUDA_VISIBLE_DEVICES=0
```

If you see Ray/Sarathi workers failing with “No CUDA GPUs are available”, double-check:

- `GSIM_CUDA_VISIBLE_DEVICES` is set
- `CUDA_VISIBLE_DEVICES` is being set (the CLI does this automatically from GSIM)

### Model assets (for the “real” run)

The “real” runner uses `scenario.model.model_ref` (a local path to weights/tokenizer). For this scenario:

- `models/llama2-7b-hf/source-data`

If this path is missing, fix it via the repo’s model bootstrap workflow (see `context/instructions/prep-dev-env.md`).

---

## What the report contains

Directory listing:

```bash
ls -la results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv
```

Expected files:

- `summary.md`: human-readable summary (inputs, profiling status, score table, embedded figures)
- `run_meta.json`: full provenance (config params, environment snapshot, git commit/dirty bit, artifact paths)
- `scores.json`: machine-readable score results (percentiles + percent errors per metric)
- `figs/`: SVG plots referenced by `summary.md`

Notes:

- `tables/` is present but may be empty depending on the run.
- The raw metrics CSVs are written under `tmp/paper_fidelity/...` and referenced from `run_meta.json`.

---

## How to interpret the results

### 1) “Normalized” metrics are per-token

Most headline metrics are **normalized by output-token count** (this matches Sarathi’s metric definitions):

- `request_e2e_time_normalized = request_e2e_time / num_output_tokens`
- `request_execution_plus_preemption_time_normalized = (execution_time + preemption_time) / num_output_tokens`
- `decode_time_execution_plus_preemption_normalized = decode_execution_plus_preemption_time / num_output_tokens`

Prefill normalized metrics are per **prompt token**:

- `prefill_time_execution_plus_preemption_normalized = prefill_execution_plus_preemption_time / num_prompt_tokens`

This normalization makes runs with different output lengths comparable (you are comparing “seconds per generated token”, etc.).

### 2) Percent error and verdicts

The scorer computes:

```
percent_error = abs(sim - real) / abs(real)
```

Verdict thresholds (see `run_meta.json` under `params.scenario.scoring.thresholds`):

- `pass`: worst percentile error ≤ 5%
- `warn`: 5–9%
- `fail`: > 9%

In this report (`summary.md`), core metrics are within a few percent:

```text
| request_execution_plus_preemption_time_normalized | p50 | ... | ... | 3.38% | pass |
| request_e2e_time_normalized                      | p50 | ... | ... | 0.91% | pass |
```

### 3) CPU overhead modeling status matters

Under `## Profiling` in `summary.md` you’ll see a CPU overhead block, e.g.:

```text
- cpu_overhead:
  - modeling: enabled
  - validation: strict
  - status: ok
  - profiled: True
```

Interpretation:

- **modeling enabled** means Vidur simulation is using the CPU overhead model (and will validate the input CSV).
- **status `ok`** means `cpu_overheads.csv` exists, is non-empty, and did not look like placeholder/dummy data.
- If status is `missing` / `placeholder` / `error`, the run is not a trustworthy “paper-fidelity” comparison (fix profiling first).

---

## Reproducing the report (step-by-step)

The exact files in this folder were generated on **2026-01-12**. If you rerun later, the report path will typically be:

- `results/reports/<YYYY-MM-DD>/paper_fidelity/llama2_7b_arxiv/`

Use `run_meta.json` to verify you reproduced the same inputs.

### Step 0: Setup

```bash
git submodule update --init --recursive
pixi install
```

Example output snippet:

```text
Submodule path 'extern/tracked/vidur': checked out ...
✨ Pixi ...
```

### Step 1: Create a host profiling root (includes CPU overhead)

Run:

```bash
pixi run paper-fidelity profile --scenario llama2_7b_arxiv --include-cpu-overhead
```

Example output (last line is the profiling root):

```text
.../tmp/paper_fidelity/profiling_roots/llama2_7b_arxiv/2026-01-12_04-06-24-491602
```

Quick verification that CPU overheads are present:

```bash
ls -la tmp/paper_fidelity/profiling_roots/llama2_7b_arxiv/2026-01-12_04-06-24-491602/data/profiling/cpu_overhead/a100_pairwise_nvlink/meta-llama/Llama-2-7b-hf/cpu_overheads.csv
head -n 5 tmp/paper_fidelity/profiling_roots/llama2_7b_arxiv/2026-01-12_04-06-24-491602/data/profiling/cpu_overhead/a100_pairwise_nvlink/meta-llama/Llama-2-7b-hf/cpu_overheads.csv
```

Example output:

```text
schedule_mean,sampler_e2e_mean,prepare_inputs_e2e_mean,model_execution_e2e_mean,...
0.0426207,0.3564672,0.1117332,15.0993323,...,meta-llama/Llama-2-7b-hf,8,1,2.33972
```

### Step 2: Run the reproduction (sim + real + score + report)

Use the profiling root you generated and enable CPU overhead modeling:

```bash
profiling_root="tmp/paper_fidelity/profiling_roots/llama2_7b_arxiv/<your-run-id>"
pixi run paper-fidelity repro \
  --scenario llama2_7b_arxiv \
  --workload static \
  --scale small \
  "scenario.vidur.profiling_root=$profiling_root" \
  "scenario.vidur.skip_cpu_overhead_modeling=false"
```

Example output (printed report directory):

```text
.../results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv
```

### Step 3: Read the summary and inspect plots

```bash
sed -n '1,120p' results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv/summary.md
ls -la results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv/figs
```

### Step 4 (optional): Regenerate `summary.md` from the JSON

If you tweak report-rendering code, regenerate without rerunning sim/real:

```bash
pixi run paper-fidelity report --dir results/reports/2026-01-12/paper_fidelity/llama2_7b_arxiv
```

---

## Troubleshooting quick hits

- CPU overhead profiling fails / produces empty CSV:
  - Check `.env`/`GSIM_CUDA_VISIBLE_DEVICES` and rerun `paper-fidelity profile --include-cpu-overhead`.
  - Inspect Ray logs under `/tmp/ray/session_latest/logs/`.
- `run_meta.json` says `git_dirty: true`:
  - Your local changes may affect reproducibility; commit or record a diff for exact provenance.
