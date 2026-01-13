# Research: Paper-fidelity more models

Date: 2026-01-13  
Feature: `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/spec.md`

## Decisions

### 1) Add one Hydra scenario YAML per paper model

- **Decision:** Add three new scenario configs under `/data1/huangzhe/code/gpu-simulate-test/configs/paper_fidelity/scenario/`:
  - `internlm_20b_arxiv.yaml`
  - `llama2_70b_arxiv.yaml`
  - `qwen_72b_arxiv.yaml`
  using `/data1/huangzhe/code/gpu-simulate-test/configs/paper_fidelity/scenario/llama2_7b_arxiv.yaml` as the template.
- **Rationale:** The existing `paper-fidelity` CLI selects scenarios by config name; adding YAML scenarios is the smallest, most consistent change and matches the existing tutorial flow.
- **Alternatives considered:**
  - Generate scenarios in Python at runtime (more moving parts; harder to audit parity-critical knobs).
  - Store model lists elsewhere (adds indirection without benefit for three models).

### 2) Reuse the Arxiv processed-lengths CSV as the baseline trace source

- **Decision:** Use `trace_source.kind=vidur_processed_lengths_csv` and reuse Vidur’s Arxiv lengths file:
  - `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur/data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv`
  with `max_tokens=4096` and a fixed seed for deterministic trace generation.
- **Rationale:** The requirements explicitly note the Sarathi paper-fidelity runner submits synthetic token IDs (model-agnostic for functional testing), so reusing a known-good lengths distribution is sufficient for end-to-end validation.
- **Alternatives considered:**
  - Create per-model, per-tokenizer prompt traces (slower, brittle, not required for “pipeline works”).
  - Require timed traces for all tests (`trace_csv`) (adds upfront work; dynamic already produces a timed trace snapshot).

### 3) Parallelism policy: configurable defaults with fail-fast on insufficient GPUs

- **Decision:** Keep scenario defaults at `tp=1, pp=1` (per clarification), but allow scenario configs (and CLI overrides) to set paper-aligned TP (InternLM-20B TP2, LLaMA2-70B TP4, Qwen-72B TP4). The matrix runner should **fail fast** when available GPUs (after `GSIM_CUDA_VISIBLE_DEVICES` pinning) cannot satisfy required parallelism, and record blocker category `insufficient GPUs`.
- **Rationale:** This enables local development on smaller hosts while still supporting paper-aligned parallelism when hardware is available, and it satisfies the explicit failure-record requirement.
- **Alternatives considered:**
  - Always relax TP to fit available GPUs (would violate the “fail fast + record insufficient GPUs” clarification).
  - Always enforce paper TP (blocks most development hosts; unnecessary for basic pipeline validation).

### 4) Host profiling is required and must include CPU overhead microbenchmarks

- **Decision:** The matrix procedure always runs host profiling per model with `paper-fidelity profile --include-cpu-overhead` and passes the resulting profiling root into repro via `scenario.vidur.profiling_root=...`.
- **Rationale:** Host profiling is required for meaningful sim-vs-real interpretation, and CPU overhead microbenchmarks are required for all models in this feature.
- **Alternatives considered:**
  - Use paper-provided profiling bundles (useful for sanity checks, but not host-matched).
  - Make CPU overhead profiling optional (conflicts with the clarified acceptance requirement).

### 5) Implement the test matrix as a Python runner (not bash-only)

- **Decision:** Provide a Python-based matrix runner (CLI wrapper or script) that orchestrates:
  - model bootstrap (optional but recommended),
  - profiling,
  - static repro (`--scale small`),
  - dynamic repro (`--scale small`),
  and produces a single manifest summarizing all attempted runs (including failures).
- **Rationale:** The existing batch script pattern (`/data1/huangzhe/code/gpu-simulate-test/scripts/run_pf_llama2_7b_sim_vs_real.sh`) uses `set -e` and stops on first failure; the new requirement needs structured failure recording and a per-matrix summary across multiple models.
- **Alternatives considered:**
  - Extend bash scripts with manual error trapping and continuation logic (possible, but harder to standardize error capture and structured output).
  - Require manual per-model command execution (does not satisfy repeatable matrix requirement).

### 6) Failure record + per-matrix manifest schema

- **Decision:** Record failures as structured JSON files and include them in a single per-matrix manifest. At minimum record:
  - attempted action (profile / repro static / repro dynamic),
  - the exact command (argv) or Hydra overrides used,
  - the error message (and traceback if available),
  - blocker category (`insufficient GPUs`, `OOM`, `missing model files`, `unsupported model`, `unknown`),
  - timestamps + provenance (git/env snapshot).
- **Rationale:** Meets FR-009 + FR-011 and enables triage without rerunning expensive GPU workloads.
- **Alternatives considered:**
  - Only write logs to stdout/stderr (not durable/discoverable).
  - Only write a manifest row without a dedicated failure payload (insufficient for debugging).

### 7) Keep reports self-contained by snapshotting inputs into report directories

- **Decision:** Continue using the existing behavior where the scoring step copies run inputs into `results/reports/.../paper_fidelity/<scenario>/inputs/` (sim/real metrics CSVs + trace/meta/capacity snapshots when available).
- **Rationale:** This already satisfies FR-007 (“portable reports not dependent on mutable tmp paths”) and should remain the source of truth for report reproducibility.
- **Alternatives considered:**
  - Keep report pointers to `tmp/` CSVs (breaks portability; tmp paths are intentionally reused).

