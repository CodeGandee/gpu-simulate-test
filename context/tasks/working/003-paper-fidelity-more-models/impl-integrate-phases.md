# Phase Integration Guide: Paper-fidelity more models

**Feature**: `003-paper-fidelity-more-models` | **Phases**: 9

## Overview

This feature extends the existing paper-fidelity workflow to additional “paper models”:

- InternLM-20B
- LLaMA2-70B
- Qwen-72B

and adds a repeatable **small-scale** (`--scale small`, 50 requests) **static+dynamic** matrix runner that:

- generates host profiling roots (including CPU overhead microbenchmarks),
- runs static + dynamic repro per model,
- writes self-contained report bundles, and
- records failures with stable blocker categorization.

**Path convention**: All repo paths are relative to `<WORKSPACE_ROOT>` (repository root).

## Phase Flow

**MUST HAVE: End-to-End Sequence Diagram**

```mermaid
sequenceDiagram
    participant U as Developer
    participant MX as paper-fidelity<br/>matrix
    participant PF as paper-fidelity<br/>profile
    participant PR as paper-fidelity<br/>repro
    participant TR as Trace builder
    participant VS as Vidur sim
    participant SR as Sarathi replay
    participant SC as Scorer
    participant FS as tmp/ +<br/>results/
    participant MN as manifest.json
    participant FR as failure_record.json

    Note over U,FS: Phase 3: scenarios (config selection)
    U->>MX: matrix --scale small<br/>--workloads static,dynamic

    loop scenarios (InternLM-20B, LLaMA2-70B, Qwen-72B)
        Note over U,FS: Phase 4: host profiling
        MX->>PF: profile --include-cpu-overhead
        alt profiling fails
            PF-->>MX: error
            MX->>FR: write failure record
        else profiling ok
            PF-->>MX: profiling_root
            loop workloads
                Note over U,FS: Phase 3/5/6: trace + sim + real
                MX->>PR: repro --workload W<br/>--scale small
                PR->>TR: generate/validate trace
                TR->>FS: tmp/.../trace.csv
                PR->>VS: run sim metrics
                VS->>FS: tmp/.../sim/request_metrics.csv
                PR->>SR: run real replay
                SR->>FS: tmp/.../real/request_metrics.csv
                PR->>SC: score + report
                SC->>FS: results/reports/.../<br/>summary.md
                alt repro fails
                    PR-->>MX: error
                    MX->>FR: write failure record
                else repro ok
                    PR-->>MX: report_dir
                end
            end
        end
    end

    Note over U,FS: Phase 7/8: per-matrix manifest + failures
    MX->>MN: write manifest.json
    MN->>FS: results/reports/.../paper_models_matrix_<run_id>/
    FS-->>U: print manifest path
```

## Artifact Flow Between Phases

```mermaid
graph TD
    subgraph P3["Phase 3: Scenarios"]
        SCFG["configs/paper_fidelity/scenario/*.yaml"]
        MODELS["models/*/source-data"]
    end

    subgraph P4["Phase 4: Profiling"]
        PROF["tmp/paper_fidelity/profiling_roots/<br/><scenario>/<timestamp>/"]
        PMETA["profiling_meta.json"]
    end

    subgraph P5P6["Phase 5/6: Repro (static+dynamic)"]
        TRACE["tmp/paper_fidelity/traces/<scenario>/trace.csv"]
        SIM["tmp/paper_fidelity/runs/<scenario>/sim/request_metrics.csv"]
        REAL["tmp/paper_fidelity/runs/<scenario>/real/request_metrics.csv"]
        CAP["tmp/paper_fidelity/runs/<scenario>/capacity/capacity.json"]
        RPT["results/reports/<DATE>/paper_fidelity/<scenario_tag>/"]
    end

    subgraph P7P8["Phase 7/8: Matrix summary"]
        MAN["paper_models_matrix_<run_id>/manifest.json"]
        FAIL["paper_models_matrix_<run_id>/failures/*.json"]
    end

    SCFG --> TRACE
    MODELS --> PROF
    PROF --> PMETA
    PROF -.->|input| SIM
    TRACE --> SIM
    TRACE --> REAL
    REAL --> CAP
    SIM --> RPT
    REAL --> RPT
    CAP --> RPT
    PMETA --> RPT
    RPT --> MAN
    FAIL --> MAN
```

## System Architecture

```mermaid
classDiagram
    class PaperFidelityCLI {
        +trace()
        +profile()
        +repro()
        +matrix()
    }

    class ScenarioValidation {
        +preflight_trace()
        +preflight_profile()
        +preflight_repro()
    }

    class PaperFidelityPaths {
        +trace_dir()
        +profiling_root_dir()
        +reports_dir()
        +matrix_dir()
    }

    class FailureRecord {
        +to_dict()
    }

    class MatrixRunner {
        +run_matrix()
    }

    class MatrixManifest {
        +write_matrix_manifest()
    }

    PaperFidelityCLI --> ScenarioValidation
    PaperFidelityCLI --> PaperFidelityPaths
    PaperFidelityCLI --> MatrixRunner
    MatrixRunner --> MatrixManifest
    MatrixRunner --> FailureRecord
```

## Use Cases

```mermaid
graph LR
    Actor((Developer))

    UC1[Generate/validate trace]
    UC2[Generate host profiling root]
    UC3[Run static repro report]
    UC4[Run dynamic repro report]
    UC5[Run paper-model matrix]
    UC6[Inspect manifest + failures]

    Actor --> UC1
    Actor --> UC2
    Actor --> UC3
    Actor --> UC4
    Actor --> UC5
    Actor --> UC6

    UC2 -.prerequisite.-> UC3
    UC2 -.prerequisite.-> UC4
    UC3 -.summarized by.-> UC6
    UC4 -.summarized by.-> UC6
    UC5 -.produces.-> UC6
```

## Activity Flow

```mermaid
stateDiagram-v2
    [*] --> ScenariosReady

    ScenariosReady --> Profiling: start matrix
    Profiling --> StaticRepro
    Profiling --> DynamicRepro

    StaticRepro --> Reported
    DynamicRepro --> Reported

    Reported --> ManifestWritten
    ManifestWritten --> [*]

    Profiling --> Failed: preflight/profiling error
    StaticRepro --> Failed: repro error
    DynamicRepro --> Failed: repro/capacity error
    Failed --> ManifestWritten: failure record saved
```

## Inter-Phase Dependencies

### Phase 3 → Phase 4/5/6 (scenario + model assets)

**Artifacts**:

- `configs/paper_fidelity/scenario/<scenario>.yaml` defines:
  - `scenario.model.model_ref` (Sarathi model assets)
  - `scenario.trace_source.path` (Vidur processed lengths CSV)
- `models/<model>/source-data` must exist for profiling + real replay

### Phase 4 → Phase 5/6 (profiling root is a required input)

**Artifacts**:

- `tmp/paper_fidelity/profiling_roots/<scenario>/<timestamp>/data/profiling/...` is passed as:
  - `scenario.vidur.profiling_root=<profiling_root>`

### Phase 5/6 → Phase 7/8 (reports + failures summarized in manifest)

**Artifacts**:

- Report bundles under `results/reports/<DATE>/paper_fidelity/<scenario_tag>/`
- Failure records under `results/reports/<DATE>/paper_fidelity/paper_models_matrix_<run_id>/failures/`

## Integration Testing

```bash
# Unit (CPU-only)
pixi run pytest tests/unit/test_env_guard.py
pixi run pytest tests/unit/test_paper_fidelity_trace.py
pixi run pytest tests/unit/test_paper_fidelity_manifest.py

# Manual (GPU required; model assets required)
pixi run paper-fidelity matrix --scale small --workloads static,dynamic --include-cpu-overhead
```

## Critical Integration Points

1. **GPU pinning is mandatory**: `GSIM_CUDA_VISIBLE_DEVICES` must be set (repo `.env` or exported).
2. **Scenario defaults vs host capacity**: defaults are `tp=1,pp=1` but matrix must fail fast when TP/PP overrides exceed visible GPUs.
3. **Report discoverability**: matrix outputs must live under a dedicated `paper_models_matrix_<run_id>/` directory, and each report must be self-contained (inputs snapshot).
4. **Failure transparency**: every failure must write a durable JSON record with a stable `blocker_category` for triage.

## References

- Individual phase guides: `context/tasks/working/003-paper-fidelity-more-models/impl-phase-*.md`
- Spec: `specs/003-paper-fidelity-more-models/spec.md`
- Plan: `specs/003-paper-fidelity-more-models/plan.md`
- Tasks breakdown (authoritative checklist): `specs/003-paper-fidelity-more-models/tasks.md`
- Data model: `specs/003-paper-fidelity-more-models/data-model.md`
- Contracts: `specs/003-paper-fidelity-more-models/contracts/`

