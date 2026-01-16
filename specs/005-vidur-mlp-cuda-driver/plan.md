# Implementation Plan: Reliable Vidur MLP profiling for driver-launched kernels

**Branch**: `[005-vidur-mlp-cuda-driver]` | **Date**: 2026-01-16 | **Spec**: `/data1/huangzhe/code/gpu-simulate-test/specs/005-vidur-mlp-cuda-driver/spec.md`  
**Input**: Feature specification from `/data1/huangzhe/code/gpu-simulate-test/specs/005-vidur-mlp-cuda-driver/spec.md`

## Summary

Fix a profiling fidelity bug where Vidur’s record-function-based MLP profiler misses GPU time when kernels are launched via the CUDA driver path, producing missing timings that currently get staged as `0.0`. The implementation will:

- Make MLP profiling method selection explicit via run configuration (no hidden code defaults).
- Improve record-function attribution to handle both runtime and driver launch paths.
- Replace silent NaN→0 staging with strict validation (default), plus an opt-in automatic fallback run.
- Enforce the same validation at both profiling-root creation and consumption.
- Preserve existing profiling-root and report directory contracts.

Planned deliverables live under `/data1/huangzhe/code/gpu-simulate-test/specs/005-vidur-mlp-cuda-driver/`.

## Technical Context

**Language/Version**: Python 3.13 (Pixi env; repo supports `>=3.11`)  
**Primary Dependencies**: Pixi, PyTorch (CUDA build), Hydra/OmegaConf, pandas, Vidur (`/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur`)  
**Storage**: Filesystem artifacts (CSV/JSON/Markdown) under `/data1/huangzhe/code/gpu-simulate-test/tmp/` and `/data1/huangzhe/code/gpu-simulate-test/results/`  
**Testing**: pytest (`pixi run pytest`) + targeted unit tests for trace attribution and validation logic  
**Target Platform**: Linux + NVIDIA GPU for profiling runs; CPU-only for unit tests (synthetic traces)  
**Project Type**: Single Python package under `/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/`  
**Performance Goals**: Trace attribution and CSV validation are linear in input size and do not materially increase profiling wall time; validation runs in <1s for typical `mlp.csv` sizes  
**Constraints**: Preserve existing profiling-root layout and report paths; strict validation by default; explicit run-config control for method/strictness/fallback; avoid changes that require committing upstream Vidur submodule history  
**Scale/Scope**: Small change set touching the Vidur profiling wrapper and profiling-root validation; adds unit tests and documentation

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The constitution at `/data1/huangzhe/code/gpu-simulate-test/.specify/memory/constitution.md` is a template placeholder (no project-specific principles/gates are defined). **Gate status: PASS** (no enforceable constitution gates).

Repo-local guardrails applied for this feature:

- Use Pixi for all commands (`pixi run ...`); do not rely on system Python.
- Keep artifacts in the established directories (`tmp/`, `results/`, profiling-root `data/profiling/...` layout).
- Add/extend pytest unit tests for new parsing/validation logic.

## Project Structure

### Documentation (this feature)

```text
/data1/huangzhe/code/gpu-simulate-test/specs/005-vidur-mlp-cuda-driver/
├── plan.md                          # This file
├── spec.md                          # Feature requirements
├── research.md                      # Phase 0 output
├── data-model.md                    # Phase 1 output
├── quickstart.md                    # Phase 1 output
├── contracts/                       # Phase 1 output
│   ├── mlp_validation_result.schema.json
│   └── profiling_meta_subset.schema.json
└── checklists/requirements.md        # Spec-quality checklist
```

### Source Code (repository root)

```text
/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/vidur_ext/
├── profile_runner.py                # staging + validation + fallback (update)
├── profiling_root.py                # consumption validation (update)
├── profiling_bundle.py              # provenance already captured (extend as needed)
├── cpu_overhead_validation.py       # existing validation pattern
├── vidur_profiling_mlp_main.py      # wrapper entrypoint (update)
└── record_function_tracer_v2.py     # new (planned)

/data1/huangzhe/code/gpu-simulate-test/tests/unit/
└── test_vidur_record_function_tracer_v2.py  # new (planned)
```

## Phase 0: Research (output: `research.md`)

Research goals for this feature:

- Confirm the Vidur trace-attribution gap (runtime vs driver launch events) and the trace fields needed for correlation.
- Confirm how MLP profiling method is selected today (Vidur default is `record_function`) and where to inject explicit selection from run configuration.
- Validate that a wrapper-level approach can improve record-function attribution without committing changes in the Vidur submodule.
- Define validation semantics and how to embed validation results into existing provenance (`profiling_meta.json`).

## Phase 1: Design & Contracts (outputs: `data-model.md`, `contracts/*`, `quickstart.md`)

Design goals:

- Define the data model for MLP validation results and provenance additions (no new artifact locations required).
- Define JSON schemas for:
  - A minimal `mlp_validation_result` object embedded into profiling meta.
  - A minimal subset of profiling meta fields used by consumers to validate compatibility/provenance.
- Document a quickstart for running profiling with explicit run-config settings using Pixi and GPU pinning.
- Run `/data1/huangzhe/code/gpu-simulate-test/.specify/scripts/bash/update-agent-context.sh codex` after plan fields are finalized.

## Phase 2: Implementation Plan (code changes; NOT executed by this command)

1. Extend run configuration to explicitly select:
   - MLP profiling method (required)
   - validation strictness (default strict)
   - opt-in automatic fallback behavior
2. Implement `RecordFunctionTracerV2` to attribute GPU time when launches occur via either runtime or driver categories, deduplicating correlation IDs to avoid double counting.
3. Replace silent NaN→0 staging with MLP CSV validation:
   - missing values always fail
   - zero-heavy signals fail in strict mode, warn in non-strict
4. Apply the same validation when consuming a profiling root, not just when staging it.
5. Record method/strictness/fallback decisions and validation summaries in existing provenance (`profiling_meta.json`).
6. Add unit tests for:
   - runtime-launch correlation attribution
   - driver-launch correlation attribution
   - validation behavior (missing vs zero-heavy, strict vs non-strict)
7. Manual verification on the known-bad scenario (LLaMA2-7B / A100 / TP=1) using Pixi:
   - confirm staged `mlp.csv` has no missing values
   - confirm the run fails fast (strict) when forced into a broken mode

## Complexity Tracking

No constitution-defined violations. Complexity to be justified during implementation (if needed): patching upstream behavior without committing submodule changes.
