# Implementation Plan: Paper-fidelity more models

**Branch**: `[003-paper-fidelity-more-models]` | **Date**: 2026-01-13 | **Spec**: `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/spec.md`
**Input**: Feature specification from `/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Add paper-fidelity scenarios for InternLM-20B, LLaMA2-70B, and Qwen-72B (excluding Qwen3-0.6B) and provide a repeatable small-scale (50 requests) static+dynamic matrix procedure that:
- generates host profiling roots (always including CPU overhead microbenchmarks),
- runs static + dynamic repro per model,
- writes discoverable, self-contained report bundles, and
- records failures + a per-matrix manifest summarizing all attempted runs.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.13 (Pixi)  
**Primary Dependencies**: Hydra (`hydra-core`), PyTorch (`torch==2.9.1+cu128`), Vidur (editable submodule), Sarathi-Serve (editable submodule), pandas/pyarrow, matplotlib/seaborn/plotly  
**Storage**: Filesystem (CSV/JSON/Markdown artifacts under `/data1/huangzhe/code/gpu-simulate-test/tmp/` and `/data1/huangzhe/code/gpu-simulate-test/results/`)  
**Testing**: pytest (via `pixi run pytest`)  
**Target Platform**: Linux x86_64 with NVIDIA GPUs (CUDA)  
**Project Type**: single (Python package + Hydra/argparse CLI)  
**Performance Goals**: Small-scale acceptance runs bounded to 50 requests; fail fast on insufficient GPUs; produce deterministic, self-contained artifacts for triage  
**Constraints**: Requires CUDA runtime/driver; real replay + profiling are GPU workflows; must record failures (attempted command, error, blocker) instead of silently failing  
**Scale/Scope**: 3 in-scope models × (profile + static repro + dynamic repro) at `--scale small` with optional configurable TP/PP per scenario

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The constitution file at `/data1/huangzhe/code/gpu-simulate-test/.specify/memory/constitution.md` is an unfilled template (placeholders), so it does not define enforceable project-specific gates.

Default gates applied for this plan:
- **GATE 1 (Scope)**: No new runtime services; keep changes within CLI/config/scripts + report artifacts. **PASS**
- **GATE 2 (Reproducibility)**: Outputs are discoverable, self-contained, and include provenance/run metadata. **PASS**
- **GATE 3 (Failure transparency)**: Failures are recorded with actionable blocker categorization. **PASS**

Post-Phase 1 design re-check: **PASS** (no additional gates introduced).

## Project Structure

### Documentation (this feature)

```text
/data1/huangzhe/code/gpu-simulate-test/specs/003-paper-fidelity-more-models/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
```text
/data1/huangzhe/code/gpu-simulate-test/
├── configs/
│   └── paper_fidelity/
│       ├── profile.yaml
│       ├── repro.yaml
│       ├── trace.yaml
│       ├── score.yaml
│       ├── scale/
│       │   ├── small.yaml
│       │   ├── medium.yaml
│       │   └── full.yaml
│       ├── workload/
│       │   ├── static.yaml
│       │   └── dynamic.yaml
│       └── scenario/
│           ├── llama2_7b_arxiv.yaml
│           ├── internlm_20b_arxiv.yaml        # new
│           ├── llama2_70b_arxiv.yaml          # new
│           └── qwen_72b_arxiv.yaml            # new
├── src/
│   └── gpu_simulate_test/
│       ├── cli/
│       │   └── paper_fidelity.py
│       ├── paper_fidelity/
│       │   ├── manifest.py
│       │   ├── paths.py
│       │   └── ...
│       └── real_bench/
│           └── backends/
│               └── sarathi_paper_fidelity_backend.py
├── models/
│   ├── internlm-20b/
│   ├── llama2-70b-hf/
│   ├── qwen-72b/
│   └── qwen3-0.6b/                            # excluded from matrix
└── tests/
    └── ...
```

**Structure Decision**: Single Python package (`/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/`) with Hydra configs in `/data1/huangzhe/code/gpu-simulate-test/configs/` and file-based artifacts under `/data1/huangzhe/code/gpu-simulate-test/tmp/` and `/data1/huangzhe/code/gpu-simulate-test/results/`.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**
No constitution violations identified for this design.
