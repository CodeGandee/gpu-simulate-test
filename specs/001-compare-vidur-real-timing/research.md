# Research: Hydra-managed experiment configs

**Feature**: Compare Vidur vs real Qwen3 A100 timing (`001-compare-vidur-real-timing`)  
**Repo root**: `/data1/huangzhe/code/gpu-simulate-test`  
**Goal**: Replace “long CLI flag lists” with composable Hydra configs while keeping Pixi-first, reproducible runs and `tmp/`-scoped artifacts.

## Decisions

### D1: Use Hydra as the primary configuration interface for all experiment entrypoints

- **Decision**: All experiment commands (`workload-spec`, `real-bench`, `vidur-profile`, `vidur-sim`, `compare-runs`) are Hydra apps.
- **Rationale**:
  - Composable config groups replace complex CLI arg surfaces.
  - Hydra snapshots the fully-resolved config into each run directory (reproducibility).
  - Sweeps/multi-runs become first-class for “try knobs, compare outputs” workflows.
- **Alternatives considered**:
  - `argparse`/`click` with many flags: hard to keep consistent across multiple commands; poor provenance.
  - Ad-hoc YAML/JSON config loading: reinvents composition/overrides, no standard run directory behavior.

### D2: Keep `configs/` as the Hydra config tree root

- **Decision**: Use repository `/data1/huangzhe/code/gpu-simulate-test/configs/` as the Hydra config root (instead of introducing a new top-level `conf/`).
- **Rationale**: `configs/` already exists and is explicitly described as “hydra-based experiments configs”; minimizing churn keeps this feature focused.
- **Alternatives considered**:
  - Rename to `conf/` to match common Hydra tutorials: rejected for unnecessary repo-wide migration during this feature.

### D3: Stage-specific output directories are owned by Hydra (`hydra.run.dir`)

- **Decision**: Each command sets `hydra.run.dir` to the desired stage directory under `tmp/`, so outputs and config snapshots are colocated:
  - Workload spec: `${repo_root}/tmp/workloads/<workload_id>/`
  - Real run: `${repo_root}/tmp/real_runs/<run_id>/`
  - Vidur sim: `${repo_root}/tmp/vidur_runs/<run_id>/`
  - Comparison: `${repo_root}/tmp/comparisons/<comparison_id>/`
  - Vidur profiling bundle: `${repo_root}/tmp/vidur_profiling/<hardware_id>/<model_id>/` (optionally timestamped)
- **Rationale**:
  - Aligns directly with the feature spec artifact layout and the repo constitution (“outputs go to `tmp/`”).
  - Makes each run directory self-contained (artifacts + config provenance).
- **Alternatives considered**:
  - A single global output root with nested stage folders managed manually: rejected because it duplicates what Hydra already does well.

### D4: Prefer `hydra.job.chdir=true` + explicit absolute paths

- **Decision**: Run each job inside its output directory (`hydra.job.chdir=true`) and ensure all input paths are absolute (or derived from `${hydra:runtime.cwd}` in config).
- **Rationale**:
  - Encourages “run dir is the world” behavior; fewer accidental writes to repo root.
  - Makes relative-output behavior consistent across stages.
- **Alternatives considered**:
  - `hydra.job.chdir=false` (run from repo root): simpler path semantics but easier to accidentally write logs/artifacts outside the run directory.

### D5: Use structured configs for type safety and config validation

- **Decision**: Define Hydra structured configs (Python `@dataclass`) for:
  - Common run metadata (model id, hardware id, git revision, timestamps)
  - Workload generation (seed, arrival process, tokenizer/model reference)
  - Real benchmark backend selection and knobs
  - Vidur profiling/sim inputs (profiling root, model id, GPU topology knobs)
- **Rationale**:
  - Prevents silent typos in overrides (`cfg.key` validation).
  - Keeps config/schema evolution readable and diffable.
- **Alternatives considered**:
  - Unstructured dict configs: rejected because this repo prioritizes “readable, typed Python”.

## Implications for the implementation plan

- **CLI UX**: Commands remain `pixi run <task>` but “configuration” moves into `configs/` presets + minimal Hydra overrides.
- **Reproducibility**: Each run directory captures:
  - Hydra-resolved config snapshot (via Hydra output)
  - `run_meta.json` (explicit, machine-readable provenance aligned with the feature spec)
- **Documentation**: `quickstart.md` should show:
  - “choose preset” (e.g., `model=qwen3_a100`) rather than “pass 15 flags”
  - where run outputs live under `tmp/`

