# Phase Integration Guide: Vidur CLI Ray runtime config

**Feature**: `006-vidur-cli-ray-config` | **Phases**: 8

## Overview

This feature makes Ray behavior reproducible and safer across host and Docker runs by letting `vidur-cli`:

- Read a small, explicitly supported set of Ray runtime settings from Hydra config (`cfg.ray.env`).
- Apply settings with per-key precedence **env > config > Ray default** (never override user-set `RAY_*`).
- Fail fast on misconfiguration (invalid values or unsupported keys) *before* starting Ray.
- Emit an “effective settings report” and persist it as `ray_settings.json` under the stage output directory.
- Optionally disable Ray for Vidur compute profiling (`profiling.compute.use_ray=false`) in supported single-GPU cases, using fallback outputs where needed.

**Path convention**: All repo paths are relative to `<WORKSPACE_ROOT>` (repository root). Run directories are under `<run_dir>` created by `vidur-cli svr init-run`.

## Phase Flow

**MUST HAVE: End-to-End Sequence Diagram**

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as vidur-cli<br/>cli/vidur_cli.py
    participant ST as stages.py
    participant SP as search_path.py
    participant RT as ray_runtime.py
    participant FS as filesystem
    participant VP as profile_runner.py
    participant RR as real_runner.py
    participant RY as Ray runtime<br/>raylet + plasma

    Note over U,FS: Phase 2: config + helper present
    U->>CLI: svr profile<br/>--run-dir run_dir
    CLI->>ST: run_profile
    ST->>SP: compose_config
    SP-->>ST: cfg (includes cfg.ray.env)
    ST->>RT: apply_ray_env_defaults
    RT-->>ST: per-key report
    ST->>FS: write profile/ray_settings.json
    opt profiling.compute.use_ray=false
        ST->>VP: run profiling<br/>no-Ray path
    end
    opt profiling.compute.use_ray=true
        ST->>VP: run profiling<br/>Ray path
        VP->>RY: ray.init
    end
    ST->>FS: update run_state.json

    Note over U,FS: Real replay stage (Sarathi backend uses Ray)
    U->>CLI: svr real<br/>--run-dir run_dir
    CLI->>ST: run_real
    ST->>SP: compose_config
    SP-->>ST: cfg (includes cfg.ray.env)
    ST->>RT: apply_ray_env_defaults
    RT-->>ST: per-key report
    ST->>FS: write real/ray_settings.json
    ST->>RR: run_token_length_replay
    RR->>RY: ray.init (Sarathi)
    ST->>FS: update run_state.json
```

## Artifact Flow Between Phases

```mermaid
graph TD
    subgraph P2["Phase 2: Foundational"]
        RCFG[configs/compare_vidur_real/ray/default.yaml]
        RHELP[src/gpu_simulate_test/ray_runtime.py]
    end

    subgraph RUN["run_dir artifacts"]
        RS[run_state.json]
        PRJ[profile/ray_settings.json]
        RRJ[real/ray_settings.json]
    end

    RCFG -->|composed into| CFG[cfg.ray.env];
    RHELP -->|applies| ENV[process env vars];
    CFG --> ENV;
    ENV --> PRJ;
    ENV --> RRJ;
    PRJ --> RS;
    RRJ --> RS;
```

## System Architecture

```mermaid
classDiagram
    class RayRuntime {
        +SUPPORTED_RAY_ENV_KEYS
        +apply_ray_env_defaults(cfg_ray_env) list
        +write_ray_settings_json(out_path, stage, settings) Path
    }

    class StageRunners {
        +run_profile(run_dir, resources, overrides) Path
        +run_real(run_dir, resources, overrides) Path
    }

    class ProfileRunner {
        +run_vidur_profiling(inputs, repo_root) Result
    }

    RayRuntime --> StageRunners: validates + reports
    StageRunners --> ProfileRunner: invokes
```

## Use Cases

```mermaid
graph LR
    User((User))

    UC1[Configure Ray settings via cfg.ray.env]
    UC2[Respect user RAY env vars]
    UC3[Disable Ray for compute profiling]
    UC4[Fail fast on misconfiguration]
    UC5[Read docs for safe Docker config]

    User --> UC1;
    User --> UC2;
    User --> UC3;
    User --> UC4;
    User --> UC5;

    UC1 -.->|enables| UC2;
    UC1 -.->|enables| UC4;
```

## Activity Flow

```mermaid
stateDiagram-v2
    [*] --> Setup
    Setup --> Foundational: configs + helper present
    Foundational --> US1: config injection + report
    US1 --> US2: env precedence
    US2 --> US3: optional no-Ray profiling
    US2 --> US4: strict validation
    US4 --> US5: docs updated
    US5 --> Polish
    Polish --> [*]
```

## Inter-Phase Dependencies

### Phase 2 → Phase 3 (Foundational → US1)

**Artifacts / config**:

- `configs/compare_vidur_real/ray/default.yaml` must exist so `cfg.ray.env` composes.
- Workflow configs must include `ray: default` in `defaults:`.

**Code dependencies**:

```python
from gpu_simulate_test.ray_runtime import apply_ray_env_defaults, write_ray_settings_json
```

### Phase 3–4 → Phase 6 (US1/US2 → US4 validation)

- US4 tightens validation rules; it must not change the output schema for `ray_settings.json`.

### Phase 3–6 → Phase 7 (Docs)

- Docs must reflect the final behavior after precedence + validation + no-Ray decisions are implemented.

## Integration Testing

```bash
cd <WORKSPACE_ROOT>

# Unit test suite (covers precedence + validation contracts):
pixi run pytest tests/unit/test_ray_runtime_config.py

# No-Ray gating tests:
pixi run pytest tests/unit/test_vidur_profile_no_ray.py

# Full unit suite (may include other unrelated tests):
pixi run pytest
```

Manual smoke docs (GPU likely required for `svr profile`):

- `tests/manual/vidur_cli_ray_settings_smoke.md`
- `tests/manual/vidur_cli_no_ray_compute_profiling_smoke.md`

## Critical Integration Points

1. **Apply settings before Ray starts**
   - `apply_ray_env_defaults()` must run before any code path that imports/starts Ray.
2. **Stable reporting**
   - `ray_settings.json` must report only env/config values (leave `effective_value=null` when default) so host vs Docker reports can be compared.
3. **No-Ray mode must not import Ray**
   - No-Ray compute profiling must avoid importing Vidur profiling entrypoints that import `ray` at module import time.

## References

- Tasks: `specs/006-vidur-cli-ray-config/tasks.md`
- Spec: `specs/006-vidur-cli-ray-config/spec.md`
- Research: `specs/006-vidur-cli-ray-config/research.md`
- Data model: `specs/006-vidur-cli-ray-config/data-model.md`
- Contracts: `specs/006-vidur-cli-ray-config/contracts/`

## Implementation Status

Implemented (all tasks completed) with key entrypoints:

- Ray config surface: `configs/compare_vidur_real/ray/default.yaml`
- Ray env apply + reporting helpers: `src/gpu_simulate_test/ray_runtime.py`
- Stage integration + run_state artifacts: `src/gpu_simulate_test/vidur_cli/stages.py`
- No-Ray compute profiling path: `src/gpu_simulate_test/vidur_ext/profile_runner.py`

Validation:

```bash
cd <WORKSPACE_ROOT>
pixi run pytest
```
