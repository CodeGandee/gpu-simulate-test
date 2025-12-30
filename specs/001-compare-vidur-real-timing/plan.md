# Implementation Plan: Compare Vidur vs real Qwen3 A100 timing

**Branch**: `001-compare-vidur-real-timing` | **Date**: 2025-12-30 | **Spec**: `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/spec.md`  
**Input**: Feature specification from `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/spec.md`

## Summary

- Deliver a Pixi-first, reproducible workflow to compare Vidur’s simulated latency distributions against real A100 timing for `Qwen/Qwen3-0.6B`, driven by a shared deterministic workload spec.
- Use Hydra to manage experiment configuration (instead of large/fragile CLI flag sets): presets live under `/data1/huangzhe/code/gpu-simulate-test/configs/`, each run snapshots its resolved config into its `tmp/` output directory, and only minimal overrides are needed for iteration and sweeps.

## Technical Context

**Language/Version**: Python 3.13 (Pixi)  
**Primary Dependencies**: `hydra-core` (configs + provenance), `torch==2.9.1+cu128` (real timing harness), `transformers` (optional real backend), `vidur` (editable submodule dependency), `pandas`/`pyarrow` (metrics I/O), `matplotlib`/`seaborn`/`plotly` (plots)  
**Storage**: Files under `/data1/huangzhe/code/gpu-simulate-test/tmp/` (CSV + JSON + Markdown + plots); no DB  
**Testing**: `pytest` for unit/integration; manual smoke scripts under `/data1/huangzhe/code/gpu-simulate-test/tests/manual/`  
**Target Platform**: Linux x86_64; NVIDIA A100 required for real runs and Vidur profiling; CPU-only is acceptable for report generation  
**Project Type**: Single Python package under `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/`  
**Performance Goals**: Support “tiny” workloads (2–10 requests) for smoke tests and moderate workloads (O(1e3) requests) for distribution comparisons  
**Constraints**: Deterministic workload generation; all timestamps in integer nanoseconds (monotonic, relative to run start); outputs default to `tmp/`; explicit `profiling_root`; early-stop handling via `num_decode_tokens_actual`; fail-fast errors on missing prerequisites; no modifications under `/data1/huangzhe/code/gpu-simulate-test/extern/`  
**Scale/Scope**: One baseline model (`Qwen/Qwen3-0.6B`), 1 GPU (TP=1/PP=1) baseline, at least 6 independently testable user stories from the spec

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Reproducibility: commands documented; Hydra configs committed; Hydra-resolved config snapshot stored per run; outputs go to `tmp/`; nondeterminism documented in `summary.md`.
- [x] Pixi env: all user-facing commands are `pixi run ...`; dependency changes (if any) update `/data1/huangzhe/code/gpu-simulate-test/pixi.lock`.
- [x] Simulator boundaries: no patches to `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur`; adapters/wrappers live in `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/`.
- [x] External assets: models/datasets remain external references under `/data1/huangzhe/code/gpu-simulate-test/models/` and `/data1/huangzhe/code/gpu-simulate-test/datasets/`; profiling data and results go to `tmp/`.
- [x] Validation: at least one manual smoke test per P1 journey is planned under `/data1/huangzhe/code/gpu-simulate-test/tests/manual/` (and targeted unit tests for schemas/determinism under `tests/unit/`).

## Project Structure

### Documentation (this feature)

```text
specs/001-compare-vidur-real-timing/
├── plan.md              # This file
├── research.md          # Hydra decisions + rationale
├── data-model.md        # Artifact entities + schema notes
├── quickstart.md        # Repro commands (Hydra-based)
├── contracts/           # File format contracts (CSV/JSON)
└── tasks.md             # Task breakdown
```

### Source Code (repository root)

```text
configs/
└── compare_vidur_real/              # Hydra config tree for this experiment
    ├── workload_spec.yaml
    ├── real_bench.yaml
    ├── vidur_profile.yaml
    ├── vidur_sim.yaml
    ├── compare_runs.yaml
    ├── model/
    ├── hardware/
    ├── workload/
    ├── backend/
    └── vidur/

src/
└── gpu_simulate_test/
    ├── cli/                         # Hydra entrypoints (one per command)
    ├── config/                      # Structured config dataclasses
    ├── workloads/                   # Workload spec generation (traces)
    ├── real_bench/                  # Real A100 timing harness
    ├── vidur_ext/                   # Vidur adapters (no submodule patches)
    └── analysis/                    # Compare + report generation

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

**Structure Decision**: Keep all project-owned code under `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/` and all Hydra presets under `/data1/huangzhe/code/gpu-simulate-test/configs/compare_vidur_real/`. All generated artifacts go under `/data1/huangzhe/code/gpu-simulate-test/tmp/` by default.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |
