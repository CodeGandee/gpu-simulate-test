# Design: `vidur-cli` (step-by-step Vidur + sim-vs-real workflows)

## Status
- **Status**: Draft (design only; no implementation in this doc)
- **Motivation**: Replace “copy/paste tutorial commands” with a first-class, step-by-step CLI that works for:
  - trace preparation
  - host profiling
  - Vidur sim runs
  - real backend runs
  - report generation
- **Primary reference workflows**:
  - Compare workflow quickstart: `specs/001-compare-vidur-real-timing/quickstart.md`
  - End-user config root (repo): `configs/compare_vidur_real/`
  - Entrypoints (stage apps): `src/gpu_simulate_test/cli/vidur_profile.py`, `src/gpu_simulate_test/cli/vidur_sim.py`, `src/gpu_simulate_test/cli/real_bench.py`, `src/gpu_simulate_test/cli/compare_runs.py`

---

## 1) Goals / non-goals

### Goals
1. **Step-by-step workflow**: expose trace / profiling / sim / real / report as separate subcommands so users can run, inspect, and rerun individual steps.
2. **Hydra-native configuration**: keep Hydra composition/overrides as the underlying configuration mechanism (no bespoke config format for experiments).
3. **Config extensibility**: allow users to add their own Hydra config directories in custom locations and compose them together with in-repo configs.
4. **Portable resource resolution**: resolve resources (models, datasets, output roots, etc.) with a clear precedence:
   - **env** > **project-local config toml** > **in-repo predefined location**
   - (repo root itself can be specified via env or project config; otherwise `pwd` is used)
5. **Reproducibility + provenance**: every step writes machine-readable metadata and points to the resolved config/resources it used.
6. **Good UX for mistakes**: when something can’t be resolved (missing model path, missing config), error messages should show *which source was tried* (env vs project config vs repo fallback) and how to fix it.

### Non-goals (for v1)
- Replace Hydra or hide Hydra entirely (users will still be able to pass `key=value` overrides).
- Implement result aggregation across many runs (explicitly deferred).
- Support remote execution / cluster orchestration (Ray/Slurm integration can be future work).
- Provide a GUI.

---

## 2) Key concepts (terminology)

### Preset keys (Hydra config groups)
The v1 design assumes `vidur-cli` uses Hydra group presets (like the current compare workflow under `configs/compare_vidur_real/`):
- `model=<key>` (e.g. `qwen3_0_6b`)
- `hardware=<key>` (e.g. `a100`)
- `backend=<key>` (e.g. `transformers`, `sarathi`)
- `workload=<key>` (trace dataset + arrival schedule inputs)
- `vidur=<key>` (Vidur-specific knobs, including replica scheduler choice + config, plus profiling root selection)

Users can add new keys by providing additional config dirs (see section 4).

### Run directory (run workspace)
`vidur-cli` should treat one directory as the “workspace” for a single run, containing:
- resolved inputs (trace + metrics CSVs)
- metadata (resolved config snapshot, resource snapshot, git info)
- comparison results and `summary.md`

Run directory contract (proposed):
- workspace root: `<pwd>/.vidur-output/<workspace-dir>/` (see section 5)
- run dir: `<workspace_root>/sim_vs_real/<run_tag>/...`

### Token-length trace dataset (the canonical dataset)
All sim-vs-real stages consume a **token-length trace file** as the canonical dataset.

Canonical schema (CSV):
- required columns:
  - `request_id` (int)
  - `arrival_time_ns` (int, relative to run start; use `0` for “burst/static”)
  - `num_prefill_tokens` (int)
  - `num_decode_tokens` (int)
- optional columns (ignored by runners, preserved for provenance):
  - `source`, `prompt_id`, `note`, ...

Common sources:
- In-repo vendored traces: `${paths.repo_root}/extern/tracked/vidur/data/processed_traces/*.csv`
  - Typically length-only (no arrivals); the CLI generates an arrival schedule deterministically.
- User-provided trace CSVs (absolute path).
  - Users may derive these from prompt corpora using their own tooling (not part of `vidur-cli` v1).

### Workload arrival kinds
Workload specs include an arrival schedule. The v1 design aligns with the existing workload generator:
- `fixed_interval` (constant `inter_arrival_ns`; includes the “burst” case where `inter_arrival_ns=0`)
- `poisson` (exponential inter-arrivals with rate `poisson_rate_per_s`)

### CPU overhead modeling (sim-vs-real parity)
For sim-vs-real runs:
- CPU overhead is **counted by default**
- If CPU overhead is disabled, `summary.md` must include a warning

---

## 3) CLI shape

### Top-level
Command name: `vidur-cli`

High-level structure:
- `vidur-cli resources ...` (resolve/inspect resources)
- `vidur-cli configs ...` (inspect available Hydra configs and where they were loaded from)
- `vidur-cli sim-vs-real ...` (primary workflow group; alias: `vidur-cli svr ...`)

### Global flags (apply to all subcommands)
These flags affect config + resource resolution and should be accepted before the subcommand name:
- `--user-config <path>`
  - Path to the `vidur-cli` TOML config.
  - Resolution order: `--user-config` > `GSIM_VIDUR_CLI_USER_CONFIG` > `<pwd>/.vidur-config/default.toml`.
- `--config-dir <path>` (repeatable)
  - Additional Hydra config directories to add (see section 4).
  - Env fallback: `GSIM_VIDUR_CLI_HYDRA_CONFIG_DIRS` (OS-specific path separator; e.g. `:` on Linux).
- `--print-resolved` (optional UX flag)
  - Print resolved repo root, config search path, and resource map before executing.

Hydra overrides:
- `vidur-cli` should accept pass-through Hydra overrides using one of these patterns:
  - **Option A**: “unknown args are overrides” (treat extra args as Hydra `key=value` overrides)
  - **Option B**: require `--` delimiter: `vidur-cli ... -- key=value other.key=value`

The design preference is **Option A** for parity with existing repo UX.

---

## 4) Hydra config composition (support user configs in custom locations)

### Requirements
1. Users can introduce new presets/configs in directories outside the repo.
2. Those custom configs can coexist with in-repo configs.
3. Users can override in-repo configs when they intentionally want to (but we should make it obvious).

### Proposed config search path model
`vidur-cli` constructs an ordered list of Hydra config roots:

1. **User-specified config dirs** (highest priority)
   - from `--config-dir ...`
   - then from `GSIM_VIDUR_CLI_HYDRA_CONFIG_DIRS`
   - then from the resolved config TOML (if it contains default config dirs)
2. **Repo config dirs** (default fallback)
   - derived from the resolved `repo_root`
   - fixed end-user root: `<repo_root>/configs/compare_vidur_real`

Notes:
- `vidur-cli` intentionally does **not** include all of `<repo_root>/configs/` in its search path. This repo contains multiple config trees grouped by development stages / workflows; only `configs/compare_vidur_real/` is considered end-user-facing for `vidur-cli` v1.
- User config dirs are expected to contain **group directories only** (`model/`, `hardware/`, `backend/`, ...). They are merged with the repo root `configs/compare_vidur_real/` by adding them earlier in the Hydra search path.

Hydra resolution rule: *the first match in the search path wins*. Therefore, user config dirs being earlier means:
- users can add new config options without conflicts
- if they reuse an existing name (e.g. `model/qwen3_0_6b.yaml`), their version will override the in-repo one.
  - The CLI MUST print a warning (user path + repo path) when this happens.

### Custom preset layout (recommended)
Recommended layout for adding custom presets that extend the compare workflow config tree:

```
<my_configs>/
  model/
    my_model.yaml
  hardware/
    my_hardware.yaml
  backend/
    my_backend.yaml
  workload/
    my_workload.yaml
  vidur/
    my_vidur.yaml
```

Then:

```bash
vidur-cli --config-dir <my_configs> svr init-run model=my_model hardware=my_hardware backend=my_backend workload=my_workload vidur=my_vidur
```

### Introspection
Add a config inspection command to prevent confusion:
- `vidur-cli configs list --group model`
  - lists available preset keys
  - prints the resolved source path for each (which config dir provided it)

---

## 5) Resource resolution (env > project config toml > repo fallback)

### Motivation
Tutorial workflows often assume:
- you are inside the repo
- models/datasets exist under `models/<name>/source-data` or `datasets/<name>/source-data`

That breaks for:
- users running from a different working directory
- users with models stored elsewhere
- future custom models/presets

### Resolution precedence
For each resource key, resolve in this order:
1. **Environment variables**
2. **Project-local config TOML**
3. **In-repo predefined location** (relative to resolved repo root)

If resolution fails, error messages should explicitly list what was tried.

### Repo root resolution
`repo_root` is the anchor for all repo-relative fallbacks.

Resolution order:
1. `GSIM_REPO_ROOT` (env var)
2. the resolved config TOML (project-local config TOML)
   - key: `resources.repo_root`
3. `pwd` (current working directory)

### Project-local config TOML
Default path: `<pwd>/.vidur-config/default.toml` (unless overridden by `GSIM_VIDUR_CLI_USER_CONFIG` or `--user-config`).

Proposed minimal schema:
```toml
[resources]
repo_root = "/abs/path/to/repo"
models_root = "/abs/path/to/models"
datasets_root = "/abs/path/to/datasets"
workspace_dir = "default" # relative -> <pwd>/.vidur-output/<workspace_dir>; absolute -> used as-is

[models]
"meta-llama/Llama-2-70b-hf" = "/abs/path/to/Llama-2-70b-hf"

[datasets]
coco2017 = "/abs/path/to/coco2017"

[hydra]
config_dirs = ["/abs/path/to/my/configs"]
```

### Resource keys
At minimum, define and resolve these keys (env var → project TOML key → repo fallback):
- `GSIM_REPO_ROOT` → `resources.repo_root` → `pwd`
- `GSIM_MODELS_ROOT` → `resources.models_root` → `<repo_root>/models`
- `GSIM_DATASETS_ROOT` → `resources.datasets_root` → `<repo_root>/datasets`
- `GSIM_VIDUR_WORKSPACE_DIR` → `resources.workspace_dir` → `default`
  - If the selected value is absolute: use it as `workspace_root`.
  - If the selected value is relative: `workspace_root = <pwd>/.vidur-output/<value>`.

Additionally, support per-asset mapping when needed:
- `models[<model_id_or_alias>] -> <path>`
- `datasets[<dataset_id_or_alias>] -> <path>`

### In-repo predefined locations (fallback)
When falling back to repo:
- models:
  - `<repo_root>/models/<alias>/source-data`
- datasets:
  - `<repo_root>/datasets/<alias>/source-data`

### How resources integrate with Hydra configs
Two supported modes (both should work; prefer Mode A):

**Mode A (recommended): `paths.*` injection**
- `vidur-cli` always injects these Hydra overrides:
  - `paths.repo_root=<resolved_repo_root>`
  - `paths.tmp_root=<workspace_root>/tmp`
  - `paths.models_root=<resolved_models_root>`
  - `paths.datasets_root=<resolved_datasets_root>`
- Preset configs should reference these `paths.*` values rather than assuming `hydra:runtime.cwd`.

**Mode B: OmegaConf resolver**
- Register `${resource:kind,key}` resolvers, e.g.:
  - `model_ref: ${resource:model,${.model_id}}`
- This makes preset YAMLs more portable, but is a bigger refactor.

For v1, Mode A is simpler and aligns with the current repo pattern (`paths.*` exists already).

---

## 6) Sim-vs-real workflow commands (step-by-step)

### Command group
`vidur-cli sim-vs-real` (alias `vidur-cli svr`)

Each subcommand should accept (either as explicit flags or via Hydra overrides):
- `--run-dir <path>` (optional but recommended)
  - If omitted, `vidur-cli` allocates one under:
    - `<workspace_root>/sim_vs_real/<generated_run_tag>/`
  - If provided as a relative path, it is interpreted relative to `workspace_root`.
- core presets:
  - `model=<key>`, `hardware=<key>`, `backend=<key>`, `workload=<key>`, `vidur=<key>`
- optional parallelism knobs (future-proofing):
  - `parallel.tp=<n>`, `parallel.pp=<n>` (or equivalent keys in the config schema)

### Proposed subcommands
1. `svr init-run`
   - Creates a run directory and writes:
     - `run_state.json` (preset keys + timestamps)
     - `resources.json` (resolved resources snapshot)
     - optional: `resolved_config.yaml` (OmegaConf resolved config)
   - This enables “step-by-step but consistent” runs.
   - Default run tag: `preset+timestamp` (see section 9); allow user override via `--run-tag <name>`.

2. `svr trace`
   - Materializes the canonical token-length trace dataset for the run:
     - `trace/trace.csv`
     - `trace/trace_meta.json`
   - Supports two paths:
     - **Import canonical trace**: accept an existing `trace.csv` that already contains `request_id,arrival_time_ns,num_prefill_tokens,num_decode_tokens`, validate schema, and copy/link it into the run dir.
     - **Build from lengths CSV**: accept a CSV that contains at least `num_prefill_tokens,num_decode_tokens` (e.g. Vidur’s `extern/tracked/vidur/data/processed_traces/*.csv`), then:
       - assign `request_id` deterministically
       - generate `arrival_time_ns` deterministically from the configured arrival schedule
   - Arrival schedule is always generated deterministically (e.g. `fixed_interval` or `poisson` with a seed), unless the imported trace already contains `arrival_time_ns`.
   - Records `trace_csv` path in `run_state.json`.

3. `svr profile`
   - Runs Vidur profiling to produce a reusable profiling root.
   - Records `profiling_root` in `run_state.json`.
   - Defaults:
     - include CPU overhead microbenchmarks by default (with `--no-include-cpu-overhead` to disable)

4. `svr sim`
   - Runs Vidur simulation using `trace/trace.csv` + `profiling_root`.
   - Vidur replica scheduler is selected/configured via the `vidur` preset (e.g. `vidur.scheduler.type=<...>` plus scheduler-specific knobs).
   - Writes a sim run directory under the run dir (or records the external location), and records `sim_run_dir` in `run_state.json`.

5. `svr real`
   - Runs the real backend using `trace/trace.csv` (no prompt dataset required at this stage).
   - The real runner uses **token-length replay**:
     - constructs synthetic inputs whose prefill length matches `num_prefill_tokens`
     - requests `num_decode_tokens` decode tokens
   - This avoids requiring the original text dataset during sim-vs-real comparisons.
   - Writes a real run directory under the run dir (or records the external location), and records `real_run_dir` in `run_state.json`.

6. `svr report`
   - Compares one real run vs one sim run and writes a report:
     - `summary.md`
     - `figs/*`
     - `tables/*` (optional)
   - Must clearly state the workload arrival kind and CPU overhead status (and warn if disabled)

### UX requirements for step-by-step
- Every step should:
  - print the primary output path (run dir or generated artifact)
  - fail fast on missing prerequisites (e.g. `svr sim` requires `trace/trace.csv` and `profiling_root`)
  - on failure, keep partial artifacts and write a `failure.json` explaining what failed

---

## 7) Examples (end-to-end; custom configs; portable resources)

### Example A: end-to-end run using in-repo presets (run from anywhere)
```bash
GSIM_REPO_ROOT=/data1/huangzhe/code/gpu-simulate-test \
GSIM_VIDUR_WORKSPACE_DIR=default \
vidur-cli svr init-run model=qwen3_0_6b hardware=a100 backend=transformers workload=default vidur=default

vidur-cli svr trace    --run-dir <printed_run_dir>
vidur-cli svr profile  --run-dir <printed_run_dir>
vidur-cli svr sim      --run-dir <printed_run_dir>
vidur-cli svr real     --run-dir <printed_run_dir>
vidur-cli svr report   --run-dir <printed_run_dir>
```

### Example B: user-defined preset config directory
```bash
vidur-cli --config-dir /home/me/my_vidur_configs svr init-run \
  model=my_model hardware=a100 backend=transformers workload=default vidur=default
```

### Example C: user-defined model location via project config
Put in `<pwd>/.vidur-config/default.toml`:
```toml
[models]
"meta-llama/Llama-2-70b-hf" = "/mnt/models/Llama-2-70b-hf"
```
Then the model preset can reference `model_id`, and the CLI resolves the local path (via `paths.*` injection or a resolver).

---

## 8) Implementation sketch (how we would build it)

### Language and packaging
- Python entrypoint `vidur-cli` installed via `pyproject.toml` console script.
- Implementation should reuse existing components from:
  - stage apps: `src/gpu_simulate_test/cli/vidur_profile.py`, `src/gpu_simulate_test/cli/vidur_sim.py`, `src/gpu_simulate_test/cli/real_bench.py`, `src/gpu_simulate_test/cli/compare_runs.py`
  - analysis/reporting: `src/gpu_simulate_test/analysis/*`
  - trace helpers (arrival schedule, CSV validation): `src/gpu_simulate_test/workloads/arrival_schedule.py`, `src/gpu_simulate_test/io.py`

### Architecture (modules)
- `gpu_simulate_test/cli/vidur_cli.py`
  - argparse top-level + subcommands
  - loads project-local config toml
  - resolves repo root + resources
  - constructs Hydra search path
  - dispatches to underlying Hydra apps or Python functions
- `gpu_simulate_test/resources/resolver.py` (new)
  - `ResourceResolver` implementing env > project toml > repo fallback
- `gpu_simulate_test/configs/search_path.py` (new)
  - builds Hydra search path from `--config-dir` + config toml + repo configs

### Backward compatibility
- Keep existing Pixi tasks working as-is (`vidur-profile`, `vidur-sim`, `real-bench`, `compare-runs`).
- `vidur-cli` can initially wrap those stage apps, then gradually move orchestration logic into shared libraries.

---

## 9) Open questions / decisions to confirm
1. **Config overriding behavior (decision)**
   - If a user-provided config file has the same name as an in-repo preset (e.g. `model/qwen3_0_6b.yaml`), the user version overrides the repo version.
   - The CLI MUST print a warning that includes both paths (user + repo) so overrides are explicit in logs.
2. **Run dir naming (decision)**
   - Default run tag is `preset+timestamp` (include `model/hardware/backend/workload/vidur` keys + UTC timestamp).
   - Allow user override (e.g. `--run-tag <name>` or `run.tag=<name>` via Hydra override).
3. **Model references (decision)**
   - Primary: configs carry `model_id` and the CLI resolves filesystem paths using resource resolution (e.g. under `GSIM_MODELS_ROOT` / `resources.models_root`).
   - Escape hatch: allow users to specify explicit filesystem paths directly in configs (e.g. `model.tokenizer_ref=/abs/path/...`) and bypass resource resolution for that model.
4. **Outputs location policy (decision)**
   - By default, all outputs are written under a workspace root:
     - `workspace_root = <pwd>/.vidur-output/<workspace-dir>` when `GSIM_VIDUR_WORKSPACE_DIR` (or `resources.workspace_dir`) is relative
     - `workspace_root = <abs path>` when `GSIM_VIDUR_WORKSPACE_DIR` (or `resources.workspace_dir`) is absolute
   - Within the workspace, all stage outputs map to subdirectories (no writes to repo `tmp/`/`results/` unless the user explicitly points there).

---

## 10) Future work (explicitly deferred)
- Multi-case aggregation into a combined report (post-sweep).
- A `vidur-cli svr sweep` command (replaces ad-hoc shell loops) plus report aggregation.
- Tools to derive token-length trace CSVs from prompt corpora (JSONL → trace), including tokenizer/version provenance capture.
- Pluggable real backends beyond Sarathi (`vllm`, `tgi`, etc.) with consistent metrics contract.
