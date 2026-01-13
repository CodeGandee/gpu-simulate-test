# Requirements: Test paper-fidelity static+dynamic runs across paper models (excluding Qwen3-0.6B)

## HEADER
- **Purpose**: Define requirements to validate that the paper-fidelity workflow (`paper-fidelity profile` + `paper-fidelity repro`) runs end-to-end for the other *paper models* in `models/` (static + dynamic), not just LLaMA2-7B.
- **Status**: Draft
- **Date**: 2026-01-13
- **In scope models** (from `models/README.md`):
  - `models/internlm-20b/` (InternLM-20B)
  - `models/llama2-70b-hf/` (LLaMA2-70B)
  - `models/qwen-72b/` (Qwen-72B)
- **Out of scope model**:
  - `models/qwen3-0.6b/` (explicitly excluded because it is not in the Vidur paper model set)
- **Primary references**:
  - Paper-fidelity CLI: `src/gpu_simulate_test/cli/paper_fidelity.py`
  - Host profiling: `src/gpu_simulate_test/paper_fidelity/profiling.py`
  - Sarathi paper-fidelity runner: `src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py`
  - Our tutorial(s):
    - How-to: `docs/tutorial/howto/tut-paper-fidelity-static-and-dynamic.md`
    - In-depth: `docs/tutorial/in-depth/adv-tut-add-paper-fidelity-scenario.md`

---

## 1) Scope and intent

### S1. What “test” means here

For each in-scope model, we want to confirm the workflow executes successfully:

1. Host-matched profiling (microbenchmarks) completes and produces a Vidur-compatible profiling root.
2. A **static** paper-fidelity repro completes and writes a report.
3. A **dynamic** paper-fidelity repro completes and writes a report (including capacity discovery).

This is a *pipeline* test (functional + integration). It is **not** a promise that the resulting sim-vs-real percent error matches the paper.

### S2. Why we can reuse the same trace lengths for multiple models (for testing)

The Sarathi paper-fidelity runner does **not** build prompts by tokenizing text; it submits synthetic `prompt_token_ids` (token id `0` repeated `num_prefill_tokens` times) and requests `max_tokens=num_decode_tokens`. This makes the trace schema model-agnostic for functional testing. See `src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py`.

---

## 2) Required test matrix

For each model, run the workflow using **small scale** (first 50 requests) to bound runtime:

- static: `--workload static --scale small`
- dynamic: `--workload dynamic --scale small`

### M1. Models to test

| Model ref | Suggested `scenario.model.model_id` | Paper TP label (reference) | Suggested `tensor_parallel_size` |
|---|---|---|---|
| `models/internlm-20b/source-data` | `internlm/internlm-20b` | InternLM-20B (TP2) | 2 |
| `models/llama2-70b-hf/source-data` | `meta-llama/Llama-2-70b-hf` | LLaMA2-70B (TP4) | 4 |
| `models/qwen-72b/source-data` | `Qwen/Qwen-72B` | Qwen-72B (TP4) | 4 |

Notes:
- The TP labels above come from the digitized paper reference rows (e.g. `context/summaries/vidur-kb/paper-results/static_fidelity_v12_request_execution_plus_preemption_time_normalized_p50.json`), but the goal here is “run works”, not “paper match”.
- If the host does not have enough GPUs for the suggested TP, the run may be blocked; record the blocker explicitly.

### M2. Hardware/environment constraints for this test

- The machine SHOULD have enough GPUs to satisfy TP requirements (2 or 4 GPUs), and `.env` SHOULD pin them via `GSIM_CUDA_VISIBLE_DEVICES=0,1,...`.
- If you only have 1 GPU available, you may still validate the *trace-only* subcommand (`paper-fidelity trace`) for the scenarios, but that does **not** satisfy this requirement document.

---

## 3) Implementation requirements

### I1. Add scenario configs per model

Add one scenario file per model under `configs/paper_fidelity/scenario/`, using `llama2_7b_arxiv.yaml` as a template:

- `internlm_20b_arxiv.yaml`
- `llama2_70b_arxiv.yaml`
- `qwen_72b_arxiv.yaml`

Each scenario MUST set:
- `name`: match the scenario key (used for artifact paths).
- `model.model_id`: as in the table above (or whatever Vidur/Sarathi require to load).
- `model.model_ref`: point to the repo model ref (e.g. `${paths.repo_root}/models/llama2-70b-hf/source-data`).
- `trace_source.kind`: use `vidur_processed_lengths_csv` (reuse Arxiv lengths file) **OR** `trace_csv` if a timed trace is preferred for the test.
- `trace_source.path`: for the simple test baseline, reuse:
  - `${paths.repo_root}/extern/tracked/vidur/data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv`
- `vidur.tensor_parallel_size` and `real.parallel.tensor_parallel_size`: set consistently (2 or 4).
- Keep parity-critical scheduler knobs explicit and aligned (chunk size, max seqs, etc.).

### I2. Testing commands must use host profiling roots

For each model/scenario, tests MUST:

1) Bootstrap model refs

```bash
bash models/<model>/bootstrap.sh
```

2) Generate a host profiling root

```bash
pixi run paper-fidelity profile --scenario <scenario> --include-cpu-overhead \
  profiling.num_gpus=<tp>
```

3) Run static and dynamic repro using that profiling root

```bash
PROFILING_ROOT="/abs/path/to/tmp/paper_fidelity/profiling_roots/<scenario>/<timestamp-dir>"

pixi run paper-fidelity repro --scenario <scenario> --workload static --scale small \
  "scenario.vidur.profiling_root=${PROFILING_ROOT}"

pixi run paper-fidelity repro --scenario <scenario> --workload dynamic --scale small \
  "scenario.vidur.profiling_root=${PROFILING_ROOT}"
```

### I3. Paper reference values are optional for this task

This task focuses on **sim-vs-real percent error**. Do not block on paper reference inclusion.

If you want to include paper reference rows in reports, you may set:

- `paper_reference.enabled=true` (global toggle)
- `scenario.paper_reference.model/trace/series` to match available rows

…but this is not required to satisfy the pipeline test.

---

## 4) Acceptance criteria

For each in-scope model, the following MUST be true:

1. `paper-fidelity profile` succeeds and writes a profiling root under:
   - `tmp/paper_fidelity/profiling_roots/<scenario.name>/<timestamp-dir>/data/profiling/...`
2. Static repro succeeds and writes a report directory containing at least:
   - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<scenario.name>/summary.md`
   - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<scenario.name>/run_meta.json`
   - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<scenario.name>/scores.json`
3. Dynamic repro succeeds and writes a report directory containing at least:
   - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<scenario.name>_dynamic_small/summary.md`
   - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<scenario.name>_dynamic_small/inputs/trace.csv`
   - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<scenario.name>_dynamic_small/inputs/capacity.json`

If any model cannot complete due to resource limits (OOM, insufficient GPUs, unsupported architecture), the run MUST still produce a short written record of:
- the exact command attempted,
- the error message,
- and the blocker category (OOM / GPU count / missing model files / Sarathi unsupported / Vidur unsupported).

