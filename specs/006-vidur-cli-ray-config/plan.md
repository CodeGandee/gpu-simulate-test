# Implementation Plan: Vidur CLI Ray runtime configuration (env > config > defaults)

**Branch**: `[006-vidur-cli-ray-config]` | **Date**: 2026-01-20 | **Spec**: `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/spec.md`
**Input**: Feature specification from `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

- Add a small, explicitly supported set of Ray runtime settings to `vidur-cli` Hydra configs and apply them with per-setting precedence **env > config > Ray defaults**.
- Validate env/config values and fail fast before any Ray-starting imports; print an effective-settings report showing value + source (env/config/default) for each supported setting.
- Add an option to disable Ray for Vidur compute profiling (default enabled) while keeping downstream workflow compatibility via fallback outputs where needed.
- Keep repo defaults opt-in: the default workflow config leaves all supported Ray settings unset unless the user configures them.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.13 (Pixi); project declares `requires-python >= 3.11`  
**Primary Dependencies**: `hydra-core`/OmegaConf (configs), `ray` (runtime), `vidur` + `sarathi` (submodules), `torch` (CUDA), `pytest`  
**Storage**: Filesystem artifacts under run directories (`tmp/`, `results/`, run_dir subfolders)  
**Testing**: `pytest` (unit + manual smoke tests under `/data1/huangzhe/code/gpu-simulate-test/tests/`)  
**Target Platform**: Linux (`linux-64` Pixi); GPU optional but typical workflows use CUDA; works in host + Docker  
**Project Type**: Single Python package (`/data1/huangzhe/code/gpu-simulate-test/src/gpu_simulate_test/`) with a console script (`vidur-cli`)  
**Performance Goals**: No measurable perf target; keep “Ray settings + validation” overhead negligible (<10ms) and avoid starting Ray when disabled for compute profiling  
**Constraints**: Must not override user-set `RAY_*`; must fail fast before any Ray-starting imports; defaults are opt-in; host/Docker runs should report identical effective settings given the same config  
**Scale/Scope**: Initial scope is 3 Ray settings + 1 “no-Ray compute profiling” option; “no-Ray” support initially limited to single-GPU case (reject unsupported configs)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The constitution file at `/data1/huangzhe/code/gpu-simulate-test/.specify/memory/constitution.md` is currently a placeholder/template (no project-specific principles filled in). For this plan we apply the template’s example principles as **provisional gates**:

- **CLI interface stability**: Any changes to `vidur-cli` must preserve existing commands and document new config knobs and precedence rules.
- **Test-first / quality gates**: Add unit tests covering env > config > defaults precedence, validation, and unsupported-key failures.
- **Integration discipline**: Keep Ray behavior control via environment variables (avoid deep patches in submodules) and ensure the settings are applied before Sarathi/Vidur imports that may start Ray.
- **Observability**: Emit an “effective settings report” (value + source per setting) for every Ray-using stage.

**Gate status (pre-Phase 0)**: PASS (no violations required for this feature).
**Gate status (post-Phase 1)**: PASS (design + contracts align with the gates; no complexity exceptions needed).

## Project Structure

### Documentation (this feature)

```text
/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks) - NOT created by this workflow
```

### Source Code (repository root)
```text
/data1/huangzhe/code/gpu-simulate-test/
├── configs/
│   └── compare_vidur_real/
│       ├── ray/                     # NEW (this feature)
│       │   └── default.yaml
│       ├── vidur_profile.yaml       # UPDATE (include ray defaults + no-Ray knob)
│       ├── real_bench.yaml          # UPDATE (include ray defaults)
│       └── vidur_sim.yaml           # UPDATE (include ray defaults; consistency)
├── src/
│   └── gpu_simulate_test/
│       ├── cli/
│       │   └── vidur_cli.py         # CLI entrypoint (already exists)
│       ├── vidur_cli/
│       │   ├── stages.py            # UPDATE (apply ray env + report before Ray usage)
│       │   └── ...
│       ├── vidur_ext/
│       │   ├── profile_runner.py    # UPDATE (no-Ray compute profiling + fallback outputs)
│       │   └── ...
│       ├── env_guard.py             # Existing env helper patterns
│       └── ray_runtime.py           # NEW helper (env > config > defaults)
└── tests/
    └── unit/
        └── test_ray_runtime_config.py  # NEW
```

**Structure Decision**: Single Python package + Hydra configs; implement the feature as a small helper module plus focused changes to `vidur-cli` stage runners and existing profiling/replay integration points.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |

## Phase 0: Research (complete before design)

Outputs:
- `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/research.md`

Research questions to resolve:
- Confirm the safest integration points to apply Ray env settings before any Ray-starting imports in both profiling and real replay code paths.
- Decide exact validation + serialization rules for each supported setting (types, allowed ranges, env var string formats).
- Decide how/where to emit the effective settings report (stdout vs file artifact) and its schema.
- Confirm the minimal supported “no-Ray compute profiling” scope and which outputs must be produced as fallbacks for downstream steps.

## Phase 1: Design & Contracts (complete after Phase 0)

Outputs:
- `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/data-model.md`
- `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/contracts/`
- `/data1/huangzhe/code/gpu-simulate-test/specs/006-vidur-cli-ray-config/quickstart.md`

Design goals:
- Keep repo defaults opt-in (no Ray env injection unless user opts in via config or env).
- Fail fast on invalid env/config values and on unsupported config keys, before importing modules that can start Ray.
- Provide an explicit, user-visible effective-settings report for every Ray-using stage.
- When disabling Ray for compute profiling, avoid starting Ray and generate downstream-compatible outputs (fallback where required) with clear indication.

## Phase 2: Implementation Planning (stop after this)

Planned work (high level):
- Add `configs/compare_vidur_real/ray/default.yaml` with nullable supported settings and documentation comments.
- Update primary workflow configs (`vidur_profile.yaml`, `real_bench.yaml`, `vidur_sim.yaml`) to include `ray: default` and add `profiling.compute.use_ray` (default `true`) where applicable.
- Implement `gpu_simulate_test.ray_runtime` helper: validate + apply env defaults + build effective-settings report.
- Integrate helper into `vidur-cli` stage runners so it runs before any imports that may start Ray.
- Implement/extend compute profiling “no-Ray” behavior in `gpu_simulate_test.vidur_ext.profile_runner` (initially single-GPU only) and ensure fallback outputs are written when skipping Ray-dependent profiling.
- Add unit tests for precedence, validation, unsupported keys, and report output format.
- Update docs to include supported settings list, precedence rules, and Docker-friendly example.
