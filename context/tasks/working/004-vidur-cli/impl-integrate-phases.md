# Phase Integration Guide: Vidur CLI

**Feature**: `004-vidur-cli` | **Phases**: 9

## Overview

This feature introduces a first-class `vidur-cli` that turns the existing sim-vs-real workflow into a resumable, step-by-step pipeline:

- Inspect resources/config roots (`resources show`, `configs list`)
- Create a run directory (`svr init-run`)
- Materialize inputs (`svr trace`)
- Run expensive stages independently (`svr profile`, `svr sim`, `svr real`)
- Generate a comparison report (`svr report`)

The run directory is the integration spine: every stage reads `run_state.json` and writes outputs under `<run_dir>/...`, preserving provenance and failures (`failure.json`) without deleting partial artifacts.

**Path convention**: All repo paths are relative to `<WORKSPACE_ROOT>` (repository root). Commands can run from arbitrary `<PWD>` via `pixi run -m <WORKSPACE_ROOT> ...`.

## Implementation Status

All phases (1–9) are implemented and tracked as complete in `specs/004-vidur-cli/tasks.md`.

- End-to-end runbook: `tests/manual/vidur_cli_smoke.md`
- “Keep green” checklist: `specs/004-vidur-cli/checklists/smoke.md`
- Artifact layout contract: `specs/004-vidur-cli/contracts/artifacts.md`

## Phase Flow

**MUST HAVE: End-to-End Sequence Diagram**

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as vidur-cli<br/>cli/vidur_cli.py
    participant RES as vidur_cli/resources.py
    participant SP as vidur_cli/search_path.py
    participant ST as vidur_cli/stages.py
    participant FS as filesystem

    Note over U,FS: Phase 3–4: Inspect
    U->>CLI: resources show
    CLI->>RES: resolve resources
    RES-->>CLI: ResourceMapV1
    CLI-->>U: prints resources

    U->>CLI: configs list<br/>--group model
    CLI->>SP: list presets
    SP-->>CLI: keys + paths
    CLI-->>U: prints list

    Note over U,FS: Phase 5: Create run dir
    U->>CLI: svr init-run<br/>model=... hardware=...<br/>backend=... workload=...<br/>vidur=...
    CLI->>ST: run_init_run
    ST->>FS: write run_state.json
    ST->>FS: write resources.json
    CLI-->>U: prints run_dir

    Note over U,FS: Phase 6: Trace
    U->>CLI: svr trace<br/>--run-dir run_dir
    CLI->>ST: run_trace
    ST->>FS: write trace/trace.csv
    ST->>FS: update run_state.json
    CLI-->>U: prints trace path

    Note over U,FS: Phase 7: Expensive stages
    U->>CLI: svr profile<br/>--run-dir run_dir
    CLI->>ST: run_profile
    ST->>FS: write profile/*
    ST->>FS: update run_state.json

    U->>CLI: svr sim<br/>--run-dir run_dir
    CLI->>ST: run_sim
    ST->>FS: write sim/request_metrics.csv
    ST->>FS: write sim/paper_fidelity/request_metrics.csv
    ST->>FS: update run_state.json

    U->>CLI: svr real<br/>--run-dir run_dir
    CLI->>ST: run_real
    ST->>FS: write real/request_metrics.csv
    ST->>FS: write real/paper_fidelity/request_metrics.csv
    ST->>FS: update run_state.json

    Note over U,FS: Phase 8: Report
    U->>CLI: svr report<br/>--run-dir run_dir
    CLI->>ST: run_report
    ST->>FS: write report/summary.md
    ST->>FS: update run_state.json
    CLI-->>U: prints report path
```

## Artifact Flow Between Phases

```mermaid
graph TD
    subgraph P5["Phase 5: init-run"]
        RS[run_state.json]
        RM[resources.json]
    end

    subgraph P6["Phase 6: trace"]
        TCSV[trace/trace.csv]
        TMETA[trace/trace_meta.json]
        TLEN[trace/trace_lengths.csv]
        TINT[trace/trace_intervals.csv]
    end

    subgraph P7["Phase 7: profile/sim/real"]
        PROOT[profile/]
        SIM[sim/request_metrics.csv]
        SIMPF[sim/paper_fidelity/request_metrics.csv]
        REAL[real/request_metrics.csv]
        REALPF[real/paper_fidelity/request_metrics.csv]
    end

    subgraph P8["Phase 8: report"]
        SUM[report/summary.md]
        TBL[report/tables/*]
        FIG[report/figs/*]
    end

    RS --> TCSV;
    RM -.->|provenance| RS;
    TCSV --> SIM;
    PROOT --> SIM;
    TCSV --> REAL;
    SIM --> SUM;
    SIMPF --> SUM;
    REAL --> SUM;
    REALPF --> SUM;
    SUM --> TBL;
    SUM --> FIG;
```

## System Architecture

```mermaid
classDiagram
    class VidurCliMain {
        +build_parser()
        +main(argv)
    }

    class ResourceResolver {
        +resolve_resources()
        +write_resources_json()
    }

    class HydraSearchPath {
        +build_config_roots()
        +discover_groups()
        +list_presets_for_group()
    }

    class RunStateStore {
        +load_run_state()
        +write_run_state()
        +write_failure_json()
    }

    class StageRunners {
        +run_init_run()
        +run_trace()
        +run_profile()
        +run_sim()
        +run_real()
        +run_report()
    }

    VidurCliMain --> ResourceResolver: resolves
    VidurCliMain --> HydraSearchPath: inspects
    VidurCliMain --> StageRunners: dispatches
    StageRunners --> RunStateStore: reads/writes
```

## Use Cases

```mermaid
graph LR
    User((User))

    UC1[Inspect resources]
    UC2[List presets]
    UC3[Initialize run dir]
    UC4[Build trace]
    UC5[Profile]
    UC6[Simulate]
    UC7[Real replay]
    UC8[Generate report]

    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5
    User --> UC6
    User --> UC7
    User --> UC8

    UC3 -.->|blocks| UC4
    UC4 -.->|blocks| UC5
    UC5 -.->|blocks| UC6
    UC4 -.->|blocks| UC7
    UC6 -.->|blocks| UC8
    UC7 -.->|blocks| UC8
```

## Activity Flow

```mermaid
stateDiagram-v2
    [*] --> ResolvedResources

    ResolvedResources --> RunInitialized: svr init-run ok
    RunInitialized --> TraceReady: svr trace ok
    TraceReady --> Profiled: svr profile ok
    Profiled --> Simulated: svr sim ok
    TraceReady --> RealRun: svr real ok
    Simulated --> ReportReady: svr report ok (needs RealRun)
    RealRun --> ReportReady: svr report ok (needs Simulated)

    TraceReady --> Failed: stage failure.json written
    Profiled --> Failed: stage failure.json written
    Simulated --> Failed: stage failure.json written
    RealRun --> Failed: stage failure.json written
    Failed --> [*]
    ReportReady --> [*]
```

## Inter-Phase Dependencies

### Phase 5 → Phase 6 (init-run → trace)

**Artifacts**:

- `<run_dir>/run_state.json` (created by `svr init-run`, updated by `svr trace`)
- `<run_dir>/resources.json` (snapshot used for provenance)

**Code dependencies**:

```python
# stages.py reads/writes the run state created in init-run
from gpu_simulate_test.vidur_cli.run_state import load_run_state, write_run_state
```

### Phase 6 → Phase 7 (trace → profile/sim/real)

**Artifacts**:

- `<run_dir>/trace/trace.csv` (canonical input)
- `<run_dir>/trace/trace_lengths.csv` + `trace_intervals.csv` (legacy compatibility for `run_vidur_sim`)

**Code dependencies**:

```python
from gpu_simulate_test.vidur_ext.sim_runner import run_vidur_sim
from gpu_simulate_test.vidur_cli.real_runner import run_token_length_replay
```

### Phase 7 → Phase 8 (sim/real → report)

**Artifacts**:

- `<run_dir>/sim/request_metrics.csv` + `token_metrics.csv`
- `<run_dir>/real/request_metrics.csv` + `token_metrics.csv`

**Code dependencies**:

```python
from gpu_simulate_test.vidur_cli.reporting import write_paper_fidelity_style_report
```

## Integration Testing

```bash
# Run from a scratch <PWD> using the repo’s Pixi env via -m:
mkdir -p /tmp/vidur-cli-e2e
cd /tmp/vidur-cli-e2e

GSIM_REPO_ROOT=<WORKSPACE_ROOT> \
RUN_DIR=$(
  pixi run -m <WORKSPACE_ROOT> vidur-cli svr init-run \
    model=qwen3_0_6b hardware=a100 backend=transformers workload=default vidur=default
)

# Deterministic trace from a tiny lengths CSV:
cat > lengths.csv <<'EOF'
num_prefill_tokens,num_decode_tokens
8,16
12,16
EOF
pixi run -m <WORKSPACE_ROOT> vidur-cli svr trace --run-dir "$RUN_DIR" --from-lengths ./lengths.csv

# Phase-7+8 stages require GPU/model assets for success; at minimum validate failure modes:
set +e
pixi run -m <WORKSPACE_ROOT> vidur-cli svr sim --run-dir "$RUN_DIR"
test -f "$RUN_DIR/failure.json"
set -e
```

## Critical Integration Points

1. **Workspace-root correctness**: `workspace_root` must never default to writing into `<WORKSPACE_ROOT>/tmp` unless explicitly configured.
2. **Run-dir resolution**: relative `--run-dir` must resolve relative to `workspace_root` (not `<PWD>`).
3. **Stage prerequisites**: missing prerequisites must be detected early and reported consistently.
4. **Artifact schema stability**: `resources.json`, `run_state.json`, and `failure.json` must keep schema v1 fields and store absolute paths.

## References

- Individual phase guides: `context/tasks/working/004-vidur-cli/impl-phase-*.md`
- Spec: `specs/004-vidur-cli/spec.md`
- Tasks breakdown: `specs/004-vidur-cli/tasks.md`
- Data model: `specs/004-vidur-cli/data-model.md`
- Contracts: `specs/004-vidur-cli/contracts/`
