<!--
Sync Impact Report
- Version change: TEMPLATE → 1.0.0
- Modified principles: N/A (initial ratification)
- Added sections: N/A (template sections filled with project-specific content)
- Removed sections: None
- Templates requiring updates:
  - ✅ updated: .specify/templates/plan-template.md
  - ✅ updated: .specify/templates/spec-template.md
  - ✅ updated: .specify/templates/tasks-template.md
  - N/A: .specify/templates/commands/ (directory not present)
- Deferred TODOs: None
-->

# gpu-simulate-test Constitution

## Core Principles

### I. Reproducibility First
- Every experiment MUST be runnable from a clean checkout using documented commands/scripts.
- All results MUST be traceable to (a) a committed config, (b) a git revision, and (c) environment metadata.
- Outputs and large artifacts MUST go to `tmp/` (or external storage); the repo keeps configs, summaries, and plots.
- Non-determinism (e.g., GPU kernels, async execution) MUST be documented alongside the experiment and captured in notes.

Rationale: This repo exists to compare simulators fairly; reproducibility is the product.

### II. Pixi-First, Pinned Environments
- Development and experiment runs MUST use the Pixi environment (`pixi run ...`) unless explicitly documented otherwise.
- Dependency changes MUST update `pixi.lock` and keep the environment reproducible on `linux-64`.
- Scripts MUST exit non-zero and print an actionable error message to stderr when required capabilities are missing (e.g., CUDA/driver).
- Avoid relying on implicit machine state (global `pip`, hidden env vars, undocumented local paths).

Rationale: A pinned, isolated environment is required for repeatable simulator comparisons.

### III. Simulator Adapters, Not Forks
- Third-party simulators live under `extern/` (git submodules) and are treated as external dependencies.
- Project-owned integration code MUST live under `src/gpu_simulate_test/` and keep simulator-specific glue isolated.
- Each simulator integration MUST have a minimal end-to-end "smoke run" (manual or integration) that produces metrics.
- Do not introduce hidden coupling between simulators (shared patches, cross-imports, or simulator-specific globals).

Rationale: Clean adapter boundaries let us compare, upgrade, and swap simulators without rewriting workloads.

### IV. External Assets Stay External
- Large models/datasets MUST NOT be committed to git; use `models/` and `datasets/` as external references.
- Bootstrap scripts MUST support configurable storage roots (e.g., `EXTERNAL_REF_ROOT`) and be safe to run repeatedly.
- Experiment outputs MUST be written to `tmp/` by default; commit only what is needed to reproduce and interpret results.

Rationale: Keeping heavyweight assets out of git preserves velocity and makes the repo portable.

### V. Readable, Typed, Tested Python
- Python changes MUST prioritize clarity and maintainability over cleverness; prefer explicit names and small modules.
- New/changed code MUST include type annotations for parameters and return values.
- Public functions/classes MUST be documented with docstrings (prefer NumPy doc style for user-facing interfaces).
- Changes MUST include at least one of: a manual test script, a unit test, or an integration test under `tests/`.

Rationale: Clear, typed code plus lightweight tests keep simulator and workload glue reliable as scope grows.

## Technical Standards

- **Repository layout**: Keep simulator/workload integrations isolated; prefer new top-level packages under `src/`.
- **Imports**: Prefer absolute imports; group standard library, third-party, then local imports.
- **Lint/typecheck**: If `ruff` and/or `mypy` are available in the Pixi env, they MUST pass for project-owned Python code.
- **Functional classes**: Use an OOP style for behavior-heavy components:
  - Prefix member variables with `m_` and initialize them in `__init__` (default to `None` where appropriate).
  - Provide read-only access via `@property`; mutate state via explicit `set_xxx()` methods with validation.
  - Keep constructors argument-free; use factory methods like `cls.from_xxx()` for initialization.
- **Data models**: Use framework-native field naming (no `m_` prefix) for compatibility with validators/serializers:
  - Default to `attrs` (`@define`, `field`, prefer `kw_only=True`) for most structured data.
  - Use `pydantic` for web request/response schemas and validation.
  - Keep business logic out of data models; use separate services/helpers for behavior.

## Workflow & Quality Gates

- Every PR MUST include:
  - A clear description of what changed and why.
  - Reproduction steps/commands (prefer `pixi run ...` commands) for any experiment or simulator integration change.
  - Notes on artifacts produced and where they are written (`tmp/`, external storage, or committed outputs).
- When adding or changing a simulator integration:
  - Document setup and pitfalls under `context/`.
  - Add/refresh an end-to-end smoke run and ensure it produces stable, parseable metrics.
- External assets:
  - Keep `models/` and `datasets/` as references only; update bootstrap docs/scripts instead of committing data.
  - Never commit large generated outputs; store them under `tmp/` and commit summaries/configs instead.

## Governance

- The constitution supersedes other guidance; project docs (`README.md`, `AGENTS.md`, `context/`) elaborate but do not override.
- Amendments MUST be made via PR and include:
  - Rationale (what problem the change solves).
  - Backward-compatibility notes (who/what breaks, and how to migrate).
  - Updates to `.specify/templates/*` when the change affects planning/spec/task expectations.
- Versioning follows semantic versioning (`MAJOR.MINOR.PATCH`):
  - **MAJOR**: removals or incompatible redefinitions of principles/governance.
  - **MINOR**: adding a new principle/section or materially expanding constraints.
  - **PATCH**: clarifications and non-semantic refinements.
- Reviewers MUST explicitly verify constitution compliance for changes that affect experiments, assets, or simulator integrations.

**Version**: 1.0.0 | **Ratified**: 2025-12-30 | **Last Amended**: 2025-12-30
