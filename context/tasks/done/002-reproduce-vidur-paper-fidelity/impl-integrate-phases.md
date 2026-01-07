# Phase Integration Guide: Reproduce Vidur paper fidelity

**Feature**: `002-reproduce-vidur-paper-fidelity` | **Phases**: 8

## Overview

This feature is a CLI-first, artifact-driven workflow to reproduce the Vidur MLSys’24 paper’s fidelity methodology. The core invariant is that **the same canonical `trace.csv`** drives both Vidur simulation and Sarathi-Serve real replay, and both sides emit **paper-aligned request metrics** with consistent definitions (especially scheduling delay and normalized latencies).

The pipeline is organized as phases so that each phase can be implemented and validated independently, while still composing into an end-to-end reproduction (`paper-fidelity repro`).

**Path convention**: All repo paths are relative to `<WORKSPACE_ROOT>` (repository root).

## Phase Flow

**MUST HAVE: End-to-End Sequence Diagram**

```mermaid
sequenceDiagram
    participant U as Contributor
    participant CLI as paper-fidelity<br/>CLI
    participant CFG as Hydra config
    participant TR as Trace builder
    participant SIM as Vidur sim
    participant CAP as Capacity search
    participant REAL as Sarathi replay
    participant SC as Scorer
    participant RP as Report writer
    participant FS as tmp/ +<br/>results/

    U->>CLI: repro --scenario S<br/>--workload static|dynamic
    CLI->>CFG: compose config<br/>(scenario/workload)
    CFG-->>CLI: cfg

    Note over U,FS: Phase 2/4: Trace
    CLI->>TR: generate/validate trace.csv
    TR->>FS: tmp/paper_fidelity/traces/S/<br/>trace.csv

    Note over U,FS: Phase 5: Sim
    CLI->>SIM: run Vidur sim<br/>(trace + profiling)
    SIM->>FS: tmp/paper_fidelity/runs/S/sim/<br/>request_metrics.csv

    alt workload=dynamic
        Note over U,FS: Phase 7: Capacity
        CLI->>CAP: discover capacity<br/>(capacity_qps, qps_85)
        CAP->>FS: tmp/paper_fidelity/runs/S/capacity/<br/>capacity.json
    end

    Note over U,FS: Phase 6: Real
    CLI->>REAL: replay trace on Sarathi<br/>(qps_85 if dynamic)
    REAL->>FS: tmp/paper_fidelity/runs/S/real/<br/>request_metrics.csv

    Note over U,FS: Phase 8: Score + Report
    CLI->>SC: score percentiles<br/>+ percent error
    CLI->>RP: write summary.md
    RP->>FS: results/reports/DATE/paper_fidelity/S/<br/>summary.md
```

## Artifact Flow Between Phases

```mermaid
graph TD
    subgraph P2["Phase 2/4: Trace"]
        TS["scenario trace source"] --> TR["trace.csv"]
        TR --> TM["trace_meta.json"]
    end

    subgraph P5["Phase 5: Sim"]
        TR --> SIM["Vidur sim"]
        PR["profiling_root"] --> SIM
        SIM --> SIMM["sim/request_metrics.csv"]
    end

    subgraph P6["Phase 6: Real"]
        TR --> REAL["Sarathi replay"]
        REAL --> REALM["real/request_metrics.csv"]
        REAL --> SEQ["sarathi/replica_0/sequence_metrics.csv"]
    end

    subgraph P7["Phase 7: Capacity"]
        REALM --> CAP["capacity search"]
        CAP --> CAPJ["capacity/capacity.json"]
    end

    subgraph P8["Phase 8: Score + Report"]
        SIMM --> SC["scoring"]
        REALM --> SC
        SC --> REP["summary.md"]
    end
```

## System Architecture

```mermaid
classDiagram
    class PaperFidelityCLI {
        +repro()
        +trace()
        +score()
    }

    class TraceBuilder {
        +read_trace_csv()
        +validate_trace()
        +generate_static()
        +generate_poisson_arrivals()
    }

    class VidurSimRunner {
        +run_vidur_paper_fidelity_sim()
    }

    class SarathiReplayRunner {
        +run_sarathi_paper_fidelity()
        +convert_sequence_metrics()
    }

    class CapacitySearcher {
        +discover_capacity()
        +is_overloaded()
    }

    class Scorer {
        +score_metric()
    }

    class ReportWriter {
        +write_summary_md()
        +diagnose_gap()
    }

    PaperFidelityCLI --> TraceBuilder
    PaperFidelityCLI --> VidurSimRunner
    PaperFidelityCLI --> SarathiReplayRunner
    PaperFidelityCLI --> CapacitySearcher
    PaperFidelityCLI --> Scorer
    PaperFidelityCLI --> ReportWriter
```

## Use Cases

```mermaid
graph LR
    Actor((Contributor))

    UC1[Generate/validate trace]
    UC2[Run Vidur sim metrics]
    UC3[Run Sarathi real replay]
    UC4[Discover capacity and qps_85]
    UC5[Score sim vs real]
    UC6[Generate report summary.md]

    Actor --> UC1
    Actor --> UC2
    Actor --> UC3
    Actor --> UC4
    Actor --> UC5
    Actor --> UC6

    UC1 -.prerequisite.-> UC2
    UC1 -.prerequisite.-> UC3
    UC3 -.prerequisite.-> UC4
    UC2 -.prerequisite.-> UC5
    UC3 -.prerequisite.-> UC5
    UC5 -.prerequisite.-> UC6
```

## Activity Flow

```mermaid
stateDiagram-v2
    [*] --> TraceReady

    TraceReady --> SimDone: run sim
    SimDone --> RealDone: run real

    RealDone --> CapacityDone: dynamic workload
    CapacityDone --> Scored

    RealDone --> Scored: static workload

    Scored --> Reported
    Reported --> [*]

    TraceReady --> TraceReady: validation failed
    SimDone --> SimDone: missing profiling bundle
    RealDone --> RealDone: missing CUDA/model assets
```

## Inter-Phase Dependencies

### Phase 2/4 → Phase 5/6 (trace is the shared contract)

**Artifacts**:

- `tmp/paper_fidelity/traces/<scenario>/trace.csv` is consumed by both Vidur and Sarathi replays.

**Code dependencies**:

```python
from gpu_simulate_test.paper_fidelity.traces import read_trace_csv
from gpu_simulate_test.vidur_ext.sim_runner import run_vidur_paper_fidelity_sim
from gpu_simulate_test.real_bench.backends.sarathi_paper_fidelity_backend import run_sarathi_paper_fidelity
```

### Phase 5/6 → Phase 8 (metrics schema must match)

**Artifacts**:

- `sim/request_metrics.csv`
- `real/request_metrics.csv`

Both must include:

- `request_scheduling_delay`
- `request_execution_plus_preemption_time_normalized`
- `request_e2e_time_normalized`

## Integration Testing

```bash
# Unit (CPU-only)
pixi run pytest tests/unit/test_paper_fidelity_trace.py
pixi run pytest tests/unit/test_paper_fidelity_vidur_metrics_schema.py
pixi run pytest tests/unit/test_paper_fidelity_real_metrics_schema.py
pixi run pytest tests/unit/test_paper_fidelity_capacity_search.py
pixi run pytest tests/test_paper_fidelity_scorer.py

# Manual (GPU requirements vary)
pixi run python tests/manual/test_paper_fidelity_trace_smoke.py
pixi run python tests/manual/test_paper_fidelity_vidur_sim_smoke.py
pixi run python tests/manual/test_paper_fidelity_real_smoke.py
pixi run python tests/manual/test_paper_fidelity_capacity_smoke.py
pixi run python tests/manual/test_paper_fidelity_repro_smoke.py
```

## Critical Integration Points

1. **Trace determinism and validity**: dynamic traces must be deterministic for a fixed seed and schema-valid for both sim and real.
2. **Metric boundary alignment**: real replay must use Sarathi’s in-engine metric definitions (avoid client-side approximations for scheduling delay).
3. **Normalized metrics preservation**: Vidur sim must preserve its normalized columns (no recomputation/renaming beyond `Request Id` → `request_id`).
4. **Capacity criterion correctness**: capacity discovery must use P99 scheduling delay > 5s (default) and record the criterion used.
5. **Overhead controls**: disable expensive tracing by default (Chrome trace, op-level metrics) so measurement overhead doesn’t dominate.

## References

- Individual phase guides: `context/tasks/done/002-reproduce-vidur-paper-fidelity/impl-phase-*.md`
- Spec: `specs/002-reproduce-vidur-paper-fidelity/spec.md`
- Data model: `specs/002-reproduce-vidur-paper-fidelity/data-model.md`
- Contracts: `specs/002-reproduce-vidur-paper-fidelity/contracts/`
- Tasks breakdown (authoritative checklist): `specs/002-reproduce-vidur-paper-fidelity/tasks.md`
