# Implementation Plan: Reproduce Vidur paper fidelity

**Branch**: `002-reproduce-vidur-paper-fidelity` | **Date**: 2026-01-05 | **Spec**: `<WORKSPACE_ROOT>/specs/002-reproduce-vidur-paper-fidelity/spec.md`  
**Input**: Feature specification from `<WORKSPACE_ROOT>/specs/002-reproduce-vidur-paper-fidelity/spec.md`

**Note**: This file was created from `.specify/templates/plan-template.md` by `<WORKSPACE_ROOT>/.specify/scripts/bash/setup-plan.sh`.

**Path convention**: `<WORKSPACE_ROOT>` refers to the repository root.

## Summary

Deliver a Pixi-first workflow to reproduce the Vidur MLSys’24 paper’s fidelity methodology on a small, representative slice (baseline scenario: LLaMA2-7B + arXiv summarization trace). The workflow must be runnable via:

- `pixi run paper-fidelity repro --scenario <scenario_name> --workload static`
- `pixi run paper-fidelity repro --scenario <scenario_name> --workload dynamic`
- `pixi run paper-fidelity score --sim <sim_metrics.csv> --real <real_metrics.csv>`

It standardizes (a) trace inputs (`trace.csv` preferred, legacy split files supported), (b) simulator and real runner outputs (`request_metrics.csv`), and (c) scoring into percentile summaries (P50/P95 minimum) plus percent error and pass/warn/fail thresholds. All heavy artifacts go to `<WORKSPACE_ROOT>/tmp/paper_fidelity/`; human-readable reports go to `<WORKSPACE_ROOT>/results/reports/<date>/paper_fidelity/<scenario_name>/summary.md`.

## Technical Context

**Language/Version**: Python 3.13 (Pixi env; repo declares `requires-python >= 3.11`)  
**Primary Dependencies**: Pixi, Hydra (`hydra-core`), Vidur (`<WORKSPACE_ROOT>/extern/tracked/vidur`), Sarathi-Serve (`<WORKSPACE_ROOT>/extern/tracked/sarathi-serve`), PyTorch (`torch==2.9.1+cu128`), pandas/pyarrow, matplotlib/seaborn/plotly  
**Storage**: Filesystem (CSV/JSON/Markdown artifacts under `<WORKSPACE_ROOT>/tmp/` and `<WORKSPACE_ROOT>/results/`)  
**Testing**: `pytest` under `<WORKSPACE_ROOT>/tests/` (unit + manual smoke tests already exist)  
**Target Platform**: `linux-64`; NVIDIA GPU required for the real replay runner (CUDA 12.8 pinned via torch `+cu128`)  
**Project Type**: Single Python package at `<WORKSPACE_ROOT>/src/gpu_simulate_test/`  
**Real Baseline Engine**: Sarathi-Serve for MVP (`<WORKSPACE_ROOT>/extern/tracked/sarathi-serve`); vLLM support is optional follow-up work  
**Trace Inputs**: Canonical `trace.csv` (preferred) with `arrived_at,num_prefill_tokens,num_decode_tokens` (optional `request_id,prompt_id`); legacy `trace_lengths.csv` + `trace_intervals.csv` supported via deterministic conversion (`arrived_at = arrival_time_ns / 1e9`)  
**Performance Goals**: Use engine-native telemetry (Sarathi/Vidur metrics stores) and disable expensive tracing by default so the wrapper adds only O(N) CSV/JSON I/O overhead under `<WORKSPACE_ROOT>/tmp/`  
**Constraints**: Reproducibility-first; deterministic seeds for trace generation and capacity search; fail-fast missing-asset checks; no large artifacts committed (use `<WORKSPACE_ROOT>/models/`, `<WORKSPACE_ROOT>/datasets/`, `<WORKSPACE_ROOT>/tmp/`)  
**Scale/Scope**: MVP supports 1 baseline scenario end-to-end (static + dynamic); extendable to multiple scenarios via configs under `<WORKSPACE_ROOT>/configs/paper_fidelity/`

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Reproducibility: commands documented; configs committed; outputs go to `<WORKSPACE_ROOT>/tmp/`; nondeterminism noted.
- [x] Pixi env: all commands use `pixi run ...`; dependency changes include `<WORKSPACE_ROOT>/pixi.lock` updates.
- [x] Simulator boundaries: changes in `<WORKSPACE_ROOT>/extern/` are justified; adapters live in `<WORKSPACE_ROOT>/src/gpu_simulate_test/`.
- [x] External assets: no large models/datasets/results committed; use `<WORKSPACE_ROOT>/models/`, `<WORKSPACE_ROOT>/datasets/`, `<WORKSPACE_ROOT>/tmp/`.
- [x] Validation: unit + manual validation is planned under `<WORKSPACE_ROOT>/tests/` (scorer fixtures + baseline scenario smoke run).

**Gate Status (pre-research)**: PASS (no violations)

**Gate Status (post-design)**: PASS

**Post-design references**:

- `<WORKSPACE_ROOT>/specs/002-reproduce-vidur-paper-fidelity/research.md`
- `<WORKSPACE_ROOT>/specs/002-reproduce-vidur-paper-fidelity/data-model.md`
- `<WORKSPACE_ROOT>/specs/002-reproduce-vidur-paper-fidelity/quickstart.md`
- `<WORKSPACE_ROOT>/specs/002-reproduce-vidur-paper-fidelity/contracts/paper_fidelity.openapi.yaml`

## Project Structure

### Documentation (this feature)

```text
<WORKSPACE_ROOT>/specs/002-reproduce-vidur-paper-fidelity/
├── plan.md              # This file
├── research.md          # Phase 0: research notes (resolves NEEDS CLARIFICATION)
├── data-model.md        # Phase 1: data + artifact model notes
├── quickstart.md        # Phase 1: runnable commands + expected artifacts
├── contracts/           # Phase 1: IO/API contracts (schemas)
└── tasks.md             # Phase 2: implementation task breakdown
```

### Source Code (repository root)

```text
<WORKSPACE_ROOT>/src/gpu_simulate_test/
├── analysis/
├── cli/
│   └── paper_fidelity.py                  # new (Hydra-driven paper-fidelity CLI)
├── paper_fidelity/                        # new
│   ├── __init__.py                        # new
│   ├── paths.py                           # new (artifact path conventions)
│   ├── traces.py                          # new (trace IO + validation; legacy compatibility)
│   ├── capacity.py                        # new (capacity discovery + 85% operating point)
│   ├── scoring.py                         # new (percentiles + percent error + thresholds)
│   └── report.py                          # new (summary.md writer + gap diagnosis)
├── real_bench/
│   └── backends/
│       └── sarathi_paper_fidelity_backend.py  # new (Sarathi replay + paper metrics)
└── vidur_ext/
    └── sim_runner.py                      # extend (paper-fidelity sim adapter)

<WORKSPACE_ROOT>/configs/
└── paper_fidelity/                       # new (defaults + scenarios + workloads)

<WORKSPACE_ROOT>/tests/
├── fixtures/paper_fidelity/              # new (small CSV fixtures)
├── manual/                               # smoke scripts (python entrypoints)
│   └── test_paper_fidelity_*_smoke.py
├── test_paper_fidelity_scorer.py         # new (fixed fixtures; unit test)
└── unit/
    └── test_paper_fidelity_*.py          # new (schema, trace, capacity unit tests)
```

**Structure Decision**: Extend the existing single-package layout by adding a paper-fidelity workflow module under `<WORKSPACE_ROOT>/src/gpu_simulate_test/` and a Hydra-driven CLI entrypoint (mirroring existing commands like `gpu_simulate_test.cli.compare_runs`). Keep simulator- and backend-specific glue isolated (Vidur under `<WORKSPACE_ROOT>/src/gpu_simulate_test/vidur_ext/`, real runner under `<WORKSPACE_ROOT>/src/gpu_simulate_test/real_bench/`).

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |

## Implementation Summary

- CLI: `pixi run paper-fidelity {trace,repro,score}` implemented by `<WORKSPACE_ROOT>/src/gpu_simulate_test/cli/paper_fidelity.py` (argparse wrapper; forwards Hydra overrides).
- Configs: `<WORKSPACE_ROOT>/configs/paper_fidelity/` provides `repro.yaml`, `trace.yaml`, `score.yaml` plus config groups `scenario/` and `workload/`.
- Artifacts: traces under `<WORKSPACE_ROOT>/tmp/paper_fidelity/traces/<scenario>/`; runs under `<WORKSPACE_ROOT>/tmp/paper_fidelity/runs/<scenario>/{sim,real,capacity}/`; reports under `<WORKSPACE_ROOT>/results/reports/<date>/paper_fidelity/<scenario>/`.
- Simulation: `<WORKSPACE_ROOT>/src/gpu_simulate_test/vidur_ext/sim_runner.py` adds a paper-fidelity sim path that preserves Vidur’s normalized metrics columns, keeps the raw Vidur output dir for debugging, and uses a per-run Vidur cache dir (`<output-dir>/vidur-cache`).
- Real replay: `<WORKSPACE_ROOT>/src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py` replays the canonical trace via Sarathi-Serve metrics store and converts to paper-fidelity `request_metrics.csv` without requiring plot tooling.
- Validation: unit tests under `<WORKSPACE_ROOT>/tests/` (`pixi run pytest tests`) + manual smoke scripts under `<WORKSPACE_ROOT>/tests/manual/` (trace/sim/real/repro).
