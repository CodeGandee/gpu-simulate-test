# Implementation Plan: Reproduce Vidur paper fidelity

**Branch**: `002-reproduce-vidur-paper-fidelity` | **Date**: 2026-01-05 | **Spec**: `/data1/huangzhe/code/gpu-simulate-test/specs/002-reproduce-vidur-paper-fidelity/spec.md`  
**Input**: Feature specification from `/data1/huangzhe/code/gpu-simulate-test/specs/002-reproduce-vidur-paper-fidelity/spec.md`

**Note**: This file was created from `.specify/templates/plan-template.md` by `/data1/huangzhe/code/gpu-simulate-test/.specify/scripts/bash/setup-plan.sh`.

## Summary

Deliver a Pixi-first workflow to reproduce the Vidur MLSys’24 paper’s fidelity methodology on a small, representative slice (baseline scenario: LLaMA2-7B + arXiv summarization trace). The workflow must be runnable via:

- `pixi run paper-fidelity repro --scenario <scenario_name> --workload static`
- `pixi run paper-fidelity repro --scenario <scenario_name> --workload dynamic`
- `pixi run paper-fidelity score --sim <sim_metrics.csv> --real <real_metrics.csv>`

It standardizes (a) trace inputs (`trace.csv` preferred, legacy split files supported), (b) simulator and real runner outputs (`request_metrics.csv`), and (c) scoring into percentile summaries (P50/P95 minimum) plus percent error and pass/warn/fail thresholds. All heavy artifacts go to `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/`; human-readable reports go to `/data1/huangzhe/code/gpu-simulate-test/results/reports/<date>/paper_fidelity/<scenario_name>/summary.md`.

## Technical Context

**Language/Version**: Python 3.13 (Pixi env; repo declares `requires-python >= 3.11`)  
**Primary Dependencies**: Pixi, Hydra (`hydra-core`), Vidur (`/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur`), Sarathi-Serve (`/data1/huangzhe/code/gpu-simulate-test/extern/tracked/sarathi-serve`), PyTorch (`torch==2.9.1+cu128`), pandas/pyarrow, matplotlib/seaborn/plotly  
**Storage**: Filesystem (CSV/JSON/Markdown artifacts under `/data1/huangzhe/code/gpu-simulate-test/tmp/` and `/data1/huangzhe/code/gpu-simulate-test/results/`)  
**Testing**: `pytest` under `/data1/huangzhe/code/gpu-simulate-test/tests/` (unit + manual smoke tests already exist)  
**Target Platform**: `linux-64`; NVIDIA GPU required for the real replay runner (CUDA 12.8 pinned via torch `+cu128`)  
**Project Type**: Single Python package at `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/`  
**Real Baseline Engine**: Sarathi-Serve for MVP (`/data1/huangzhe/code/gpu-simulate-test/extern/tracked/sarathi-serve`); vLLM support is optional follow-up work  
**Trace Inputs**: Canonical `trace.csv` (preferred) with `arrived_at,num_prefill_tokens,num_decode_tokens` (optional `request_id,prompt_id`); legacy `trace_lengths.csv` + `trace_intervals.csv` supported via deterministic conversion (`arrived_at = arrival_time_ns / 1e9`)  
**Performance Goals**: Use engine-native telemetry (Sarathi/Vidur metrics stores) and disable expensive tracing by default so the wrapper adds only O(N) CSV/JSON I/O overhead under `/data1/huangzhe/code/gpu-simulate-test/tmp/`  
**Constraints**: Reproducibility-first; deterministic seeds for trace generation and capacity search; fail-fast missing-asset checks; no large artifacts committed (use `/data1/huangzhe/code/gpu-simulate-test/models/`, `/data1/huangzhe/code/gpu-simulate-test/datasets/`, `/data1/huangzhe/code/gpu-simulate-test/tmp/`)  
**Scale/Scope**: MVP supports 1 baseline scenario end-to-end (static + dynamic); extendable to multiple scenarios via configs under `/data1/huangzhe/code/gpu-simulate-test/configs/paper_fidelity/`

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Reproducibility: commands documented; configs committed; outputs go to `/data1/huangzhe/code/gpu-simulate-test/tmp/`; nondeterminism noted.
- [x] Pixi env: all commands use `pixi run ...`; dependency changes include `/data1/huangzhe/code/gpu-simulate-test/pixi.lock` updates.
- [x] Simulator boundaries: changes in `/data1/huangzhe/code/gpu-simulate-test/extern/` are justified; adapters live in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/`.
- [x] External assets: no large models/datasets/results committed; use `/data1/huangzhe/code/gpu-simulate-test/models/`, `/data1/huangzhe/code/gpu-simulate-test/datasets/`, `/data1/huangzhe/code/gpu-simulate-test/tmp/`.
- [x] Validation: unit + manual validation is planned under `/data1/huangzhe/code/gpu-simulate-test/tests/` (scorer fixtures + baseline scenario smoke run).

**Gate Status (pre-research)**: PASS (no violations)

**Gate Status (post-design)**: PASS

**Post-design references**:

- `/data1/huangzhe/code/gpu-simulate-test/specs/002-reproduce-vidur-paper-fidelity/research.md`
- `/data1/huangzhe/code/gpu-simulate-test/specs/002-reproduce-vidur-paper-fidelity/data-model.md`
- `/data1/huangzhe/code/gpu-simulate-test/specs/002-reproduce-vidur-paper-fidelity/quickstart.md`
- `/data1/huangzhe/code/gpu-simulate-test/specs/002-reproduce-vidur-paper-fidelity/contracts/paper_fidelity.openapi.yaml`

## Project Structure

### Documentation (this feature)

```text
/data1/huangzhe/code/gpu-simulate-test/specs/002-reproduce-vidur-paper-fidelity/
├── plan.md              # This file
├── research.md          # Phase 0: research notes (resolves NEEDS CLARIFICATION)
├── data-model.md        # Phase 1: data + artifact model notes
├── quickstart.md        # Phase 1: runnable commands + expected artifacts
├── contracts/           # Phase 1: IO/API contracts (schemas)
└── tasks.md             # Phase 2: implementation task breakdown
```

### Source Code (repository root)

```text
/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/
├── analysis/
├── cli/
│   └── paper_fidelity.py          # new
├── paper_fidelity/                # new
│   ├── traces.py                  # new (trace IO + validation; legacy compatibility)
│   ├── capacity.py                # new (capacity discovery + 85% operating point)
│   ├── scoring.py                 # new (percentiles + percent error + thresholds)
│   └── report.py                  # new (summary.md writer; reuses analysis/report patterns)
├── real_bench/
│   └── backends/
│       └── sarathi_backend.py     # extend (paper metrics timestamps)
└── vidur_ext/
    └── sim_runner.py              # extend (preserve required paper metric columns)

/data1/huangzhe/code/gpu-simulate-test/configs/
├── paper_fidelity/                # new (scenarios + defaults)
└── compare_vidur_real/
    └── backend/                   # extend (real backend alignment knobs)

/data1/huangzhe/code/gpu-simulate-test/tests/
└── test_paper_fidelity_scorer.py  # new (fixed fixtures; unit test)
```

**Structure Decision**: Extend the existing single-package layout by adding a paper-fidelity workflow module under `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/` and a Hydra-driven CLI entrypoint (mirroring existing commands like `gpu_simulate_test.cli.compare_runs`). Keep simulator- and backend-specific glue isolated (Vidur under `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_ext/`, real runner under `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/real_bench/`).

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |
