# Implementation Plan: Compare Vidur vs real Qwen3 A100 timing

**Branch**: `001-compare-vidur-real-timing` | **Date**: 2025-12-30 | **Spec**: `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/spec.md`  
**Input**: Feature specification from `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/spec.md`

**Note**: This file is created from this template by `.specify/scripts/bash/setup-plan.sh`.

## Summary

Build a reproducible, trace-driven workflow to compare Vidur’s simulated latency distributions against real A100 timing for `Qwen/Qwen3-0.6B` under the same workload spec (token lengths + arrival schedule), producing standardized artifacts and a report with P50/P90/P99 + CDF plots for TTFT and per-token latency.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.13 (Pixi)  
**Primary Dependencies**: `pixi`, `torch==2.9.1+cu128`, `transformers`, `pandas`, `matplotlib`, Vidur (`extern/tracked/vidur` editable), Sarathi-Serve (`extern/tracked/sarathi-serve` editable)  
**Storage**: Files under `/data1/huangzhe/code/gpu-simulate-test/tmp/` (CSV + JSON + Markdown reports)  
**Testing**: `pytest` (planned), plus manual smoke scripts under `/data1/huangzhe/code/gpu-simulate-test/tests/manual/`  
**Target Platform**: Linux (`linux-64`) with NVIDIA GPU (A100 for ground-truth timing)  
**Project Type**: Single Python package under `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/`  
**Performance Goals**: Support tiny workloads for smoke (2–10 req) and typical experiment workloads (10^2–10^4 req) with analysis/report generation dominated by CSV processing time.  
**Constraints**: Reproducible from clean checkout; no large artifacts committed; simulator adapters live in `src/gpu_simulate_test/`; default outputs in `tmp/`.  
**Scale/Scope**: Single-model/single-GPU baseline: `Qwen/Qwen3-0.6B` on A100, TP=1/PP=1, comparing TTFT + per-token latency distributions (early-stop aware).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Reproducibility: commands documented; configs committed; outputs go to `tmp/`; nondeterminism noted.
- [x] Pixi env: all commands use `pixi run ...`; dependency changes include `pixi.lock` updates.
- [x] Simulator boundaries: changes in `extern/` are justified; adapters live in `src/gpu_simulate_test/`.
- [x] External assets: no large models/datasets/results committed; use `models/`, `datasets/`, `tmp/` patterns.
- [x] Validation: at least one of manual/unit/integration coverage is planned under `tests/`.

## Project Structure

### Documentation (this feature)

```text
specs/001-compare-vidur-real-timing/
├── plan.md              # This file
├── research.md          # Optional: research notes
├── data-model.md        # Optional: data model notes
├── quickstart.md        # Optional: quickstart for the feature
├── contracts/           # Optional: IO/API contracts
└── tasks.md             # Task breakdown
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
src/
└── gpu_simulate_test/

tests/
├── integration/
├── manual/
└── unit/

context/
scripts/
models/
datasets/
extern/
tmp/
```

**Structure Decision**: Add new modules under `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/` for workload generation, real benchmarking, Vidur adapters, and comparison/reporting; add minimal tests under `/data1/huangzhe/code/gpu-simulate-test/tests/` (manual smoke + unit schema checks).

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |

## Phase 0: Outline & Research

Deliverable: `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/research.md`

- Confirm Sarathi-Serve feasibility for Qwen3 (current summary indicates Qwen3 is not registered in Sarathi; plan for `transformers` backend as the Qwen3 ground-truth path, with Sarathi used for supported-model smoke and/or later extension).
- Identify Vidur CLI flags needed to point profiling inputs at a user-provided profiling root (avoid `./data/profiling/...` CWD coupling).
- Decide canonical file schemas (CSV columns + JSON fields) for:
  - Workload spec (`prompts.jsonl`, `trace_lengths.csv`, `trace_intervals.csv`, `workload_meta.json`)
  - Run outputs (`run_meta.json`, `request_metrics.csv`, `token_metrics.csv`)
- Decide timing measurement strategy for real runs (monotonic `*_ns` fields; token timestamps captured to `token_metrics.csv`).

## Phase 1: Design & Contracts

Deliverables:

- `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/data-model.md`
- `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/contracts/`
- `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/quickstart.md`

Outputs:

- Data model for the workload/run artifacts (entities, required fields, relationships).
- “Contracts” for the CSV/JSON formats (schemas and required columns).
- Quickstart that runs a tiny end-to-end flow (workload → real bench → Vidur sim → compare) writing to `tmp/`.
- Update agent context via `/data1/huangzhe/code/gpu-simulate-test/.specify/scripts/bash/update-agent-context.sh codex` once plan fields are finalized.

## Phase 2: Task Breakdown (Implementation)

Deliverable: `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/tasks.md`

- Implement workload generator CLI (`workload-spec`) producing canonical traces in `tmp/workloads/<workload_id>/`.
- Implement real benchmark CLI (`real-bench`) with `--backend sarathi|transformers` producing `tmp/real_runs/<run_id>/`.
- Implement Vidur adapters:
  - `vidur-profile` to generate/stage profiling inputs under an explicit profiling root.
  - `vidur-sim` wrapper to run Vidur from repo root with profiling paths pointing at the profiling root.
- Implement comparator (`compare-runs`) producing `tmp/comparisons/<run_id>/summary.md` + plots.
- Add validation:
  - Manual smoke script for tiny workload end-to-end.
  - Unit tests for schema/column validation and determinism (seeded workload generation).
