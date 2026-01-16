# Phase Integration Guide: Reliable Vidur MLP profiling for driver-launched kernels

**Feature**: `005-vidur-mlp-cuda-driver` | **Phases**: 6

## Overview

This feature fixes a profiling fidelity failure mode where Vidur’s record-function-based MLP profiler can miss GPU time when kernels are launched via the CUDA driver path (`cuda_driver`), producing missing timings that were previously masked as `0.0` during staging.

The integrated solution has three pillars:

1. **Explicit per-run method selection** (`profiling.mlp.profile_method`) to remove hidden defaults.
2. **Robust attribution** for record-function profiling via `RecordFunctionTracerV2` (counts both runtime and driver launch paths).
3. **Strict-by-default validation** at both staging and consumption, with an opt-in automatic fallback method for recovery.

**Path convention**: All repo paths are relative to `<WORKSPACE_ROOT>` (repository root).

## Implementation Status

Planned (implementation guides and tasks are authored; code changes are not yet applied).

- Tasks: `specs/005-vidur-mlp-cuda-driver/tasks.md`
- Guides: `context/tasks/working/005-vidur-mlp-cuda-driver/impl-phase-*.md`

## Phase Flow

**MUST HAVE: End-to-End Sequence Diagram**

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as EntryPoint<br/>(Hydra)
    participant PR as profile_runner.py
    participant MLP as vidur_profiling_mlp_main.py
    participant TR as RecordFunctionTracerV2<br/>(record_function)
    participant VAL as mlp_validation.py
    participant FS as filesystem
    participant ROOT as profiling_root.py
    participant SIM as sim_runner.py

    Note over U,FS: Phase 2: Config must be explicit
    U->>CLI: set profiling.mlp.profile_method

    Note over U,FS: Phase 3: MLP profiling + staging
    CLI->>PR: run_vidur_profiling(inputs)
    PR->>MLP: subprocess --profile_method A
    alt A = record_function
        MLP->>TR: patched tracer used
        TR-->>FS: profiler_trace_*.json
        TR-->>MLP: time_stats dict
    else A != record_function
        MLP-->>MLP: Vidur timer stats
    end
    MLP-->>FS: raw mlp.csv
    PR->>FS: stage data/profiling/.../mlp.csv

    Note over U,FS: Phase 4: Validation + optional fallback
    PR->>VAL: validate_mlp_csv
    alt valid
        VAL-->>PR: MlpValidationResult
    else invalid
        alt fallback enabled
            PR->>MLP: subprocess --profile_method B
            MLP-->>FS: raw mlp.csv
            PR->>FS: re-stage mlp.csv
            PR->>VAL: validate_mlp_csv
            VAL-->>PR: ok
        else fallback disabled
            PR-->>CLI: raise (remediation)
            CLI-->>U: fail fast
        end
    end
    PR-->>CLI: profiling_root ready
    CLI-->>FS: write profiling_meta.json<br/>(includes mlp_validation)

    Note over U,FS: Phase 4 (consumption): validate on load
    U->>SIM: run simulation/report
    SIM->>ROOT: validate_profiling_root
    ROOT->>VAL: validate_mlp_csv
    alt strict ok
        ROOT-->>SIM: ok
    else strict fail
        ROOT-->>SIM: error
        SIM-->>U: stop with message
    end
```

## Artifact Flow Between Phases

```mermaid
graph TD
    subgraph P3["Phase 3: profiling outputs"]
        RAW[cache_dir/mlp/.../mlp.csv]
        TRACE[cache_dir/profiler_traces/*.json]
    end

    subgraph P3S["Phase 3: staged profiling root"]
        MLP[data/profiling/compute/<device>/<model>/mlp.csv]
        ATTN[data/profiling/compute/<device>/<model>/attention.csv]
        META[profiling_meta.json]
    end

    subgraph P4C["Phase 4: consumers"]
        SIMRUN[vidur sim / report]
    end

    RAW --> MLP;
    TRACE -.->|attribution evidence| RAW;
    MLP --> META;
    ATTN --> META;
    META -.->|provenance| SIMRUN;
    MLP -.->|validated on load| SIMRUN;
```

## System Architecture

```mermaid
classDiagram
    class VidurProfileInputs {
        +model_id: str
        +hardware_id: str
        +profiling_root: Path
        +mlp_profile_method: str
        +mlp_validation_mode: str
        +mlp_small_input_threshold: int
        +mlp_zero_heavy_limit: float
        +mlp_fallback_enabled: bool
        +mlp_fallback_method: str
    }

    class RecordFunctionTracerV2 {
        +__enter__()
        +__exit__(*args)
        +get_operation_time_stats() dict
    }

    class MlpValidationResult {
        +missing_cells_total: int
        +missing_columns: list
        +zero_heavy_columns: list
        +warnings: list
        +as_jsonable() dict
    }

    class ProfilingRootLayout {
        +profiling_root: Path
        +device: str
        +model_id: str
        +mlp_validation_mode: str
        +mlp_small_input_threshold: int
        +mlp_zero_heavy_limit: float
    }

    VidurProfileInputs --> RecordFunctionTracerV2: record_function path
    VidurProfileInputs --> MlpValidationResult: staging validation
    ProfilingRootLayout --> MlpValidationResult: consumption validation
```

## Use Cases

```mermaid
graph LR
    User((User))

    UC1[Setup env + submodules]
    UC2[Run profiling with explicit method]
    UC3[Validate + fallback on failure]
    UC4[Consume profiling root safely]

    User --> UC1
    User --> UC2
    UC2 --> UC3
    UC3 --> UC4
```

## Activity Flow

```mermaid
stateDiagram-v2
    [*] --> SetupReady

    SetupReady --> ConfigReady: profile_method explicit
    ConfigReady --> ProfileRunning
    ProfileRunning --> Staged: mlp.csv staged
    Staged --> Validated: validation ok
    Staged --> Failed: strict validation fail
    Failed --> ProfileRunning: fallback enabled
    Validated --> ProfilingRootReady
    ProfilingRootReady --> Consumed: validate on load
    Consumed --> [*]
```

## Inter-Phase Dependencies

### Phase 2 → Phase 3 (Foundational → US1)

**Artifacts / Inputs**:

- Hydra configs must include `profiling.mlp.profile_method` (required).

**Code dependencies**:

```python
from gpu_simulate_test.vidur_ext.profile_runner import VidurProfileInputs, run_vidur_profiling
```

### Phase 3 → Phase 4 (US1 → US2)

**Artifacts / Inputs**:

- Staged `data/profiling/compute/.../mlp.csv` (must be non-missing for core targets).

**Code dependencies**:

```python
from gpu_simulate_test.vidur_ext.mlp_validation import validate_mlp_csv
```

### Phase 4 → Phase 5 (US2 → US3)

**Artifacts / Inputs**:

- Stable public interfaces (`RecordFunctionTracerV2`, `validate_mlp_csv`) that unit tests can target.

## Integration Testing

```bash
cd <WORKSPACE_ROOT>

# Config-only sanity (no GPU required):
pixi run python -m gpu_simulate_test.cli.vidur_profiling_bundle --cfg job --resolve \
  output.dir=/tmp/vidur-profiling-bundle-cfg \
  profiling.mlp.profile_method=cuda_event

# Full profiling requires a CUDA-capable GPU; follow:
# specs/005-vidur-mlp-cuda-driver/quickstart.md
```

## References

- Phase guides: `context/tasks/working/005-vidur-mlp-cuda-driver/impl-phase-*.md`
- Spec: `specs/005-vidur-mlp-cuda-driver/spec.md`
- Tasks: `specs/005-vidur-mlp-cuda-driver/tasks.md`
- Data model: `specs/005-vidur-mlp-cuda-driver/data-model.md`
- Contracts: `specs/005-vidur-mlp-cuda-driver/contracts/`

