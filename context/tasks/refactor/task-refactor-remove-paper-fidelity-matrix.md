---
description: "Refactor plan: remove `paper-fidelity matrix`, replace with shell sweep + aggregated combined report"
created: 2026-01-13
---

# Refactor Plan: Remove `paper-fidelity matrix` (use `.sh` sweeps; aggregation deferred)

## What to Refactor

- Remove the `matrix` subcommand from the `paper-fidelity` CLI:
  - `src/gpu_simulate_test/cli/paper_fidelity.py` (argparse wiring + execution path)
  - `src/gpu_simulate_test/paper_fidelity/matrix.py` (orchestration logic)
  - `src/gpu_simulate_test/paper_fidelity/matrix_manifest.py` (matrix manifest schema/output)
  - `src/gpu_simulate_test/paper_fidelity/paper_models.py` (paper-model scenario set), if it becomes unused
- Replace “run a sweep” with:
  - One or more shell scripts under `scripts/` that run `profile` + `repro` per case (scenario × workload × scale)
  - (Future) an aggregation step that reads per-case outputs and produces a combined report directory

## Why Refactor

- **Flexibility**: `.sh` sweeps can pass Hydra overrides per run (TP/PP knobs, scheduler knobs, profiling knobs) without being blocked by `matrix`’s “no Hydra overrides” constraint.
- **Simplicity**: `matrix` duplicates orchestration logic that is easier to inspect/debug in a shell script with explicit commands.
- **Reproducibility**: (future) aggregation can be rerun cheaply from saved `results/reports/.../paper_fidelity/*` without recomputing the expensive GPU work.
- **Surface area**: removing `matrix` reduces CLI/API surface that must be maintained and documented.

## How to Refactor

### Step 0: Define the new user-facing workflow

Target flow:
1. Run a sweep via `scripts/paper_fidelity_sweep.sh` (or similarly named) which emits per-case reports under `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<scenario_tag>/`.
2. (Deferred) Add an aggregation step that summarizes the sweep into a combined report directory.

Notes:
- Keep using the existing per-run artifacts:
  - Success: `summary.md`, `run_meta.json`, `scores.json`, `inputs/*` under each report dir
  - Failure: `failure_record.json` written into the report dir on `repro` failures; profiling failures write `failure_record.json` into the profiling root under `tmp/paper_fidelity/profiling_roots/...`

### Step 1: Add a sweep shell script (no Python orchestration)

Create `scripts/paper_fidelity_sweep.sh` that:
- Accepts: `--run-id`, `--date` (optional), `--scale`, `--scenarios`, `--workloads`, `--include-cpu-overhead`, plus **global parallelism flags** `--tp` and `--pp` that apply to all cases
- For each case:
  - Runs `paper-fidelity profile` (captures the printed profiling root path)
  - Runs `paper-fidelity repro` for static and/or dynamic using `scenario.vidur.profiling_root=<captured>`
- Writes a machine-readable “sweep log” (suggested) to a stable location, e.g.:
  - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/sweep_<run_id>/cases.jsonl`
  - Each line contains: scenario_key, workload, scale, command argv, profiling_root, report_dir or failure_record path

Design detail: the `.sh` script is the right place to encode “matrix variants”, e.g.:
- per-scenario TP/PP overrides (and matching profiling `num_gpus`/`tensor_parallel_size`)
- per-scenario `max_num_seqs` or chunk sizing for OOM mitigation

**Global TP/PP behavior (required)**

The sweep script should support setting TP/PP for *all* cases via `--tp` and `--pp`, and apply those consistently across the profiling + repro commands:

- Real replay parallelism:
  - `scenario.real.parallel.tensor_parallel_size=${TP}`
  - `scenario.real.parallel.pipeline_parallel_size=${PP}`
- Vidur sim parallelism:
  - `scenario.vidur.tensor_parallel_size=${TP}`
  - `scenario.vidur.num_pipeline_stages=${PP}`
- Profiling parallelism:
  - `profiling.tensor_parallel_size=${TP}`
  - `profiling.num_gpus=$((TP * PP))`

This global setting is meant to make “fit-to-host” sweeps easy (e.g., force TP=1/PP=1 on a 1-GPU host).
Per-scenario overrides (if still needed) can be layered on top inside the script as special-cases.

### Step 2: Remove `matrix` subcommand and its implementation

- Delete CLI wiring in `src/gpu_simulate_test/cli/paper_fidelity.py`:
  - `sub.add_parser("matrix") ...`
  - `elif args.cmd == "matrix": ...`
- Remove (or archive) now-dead code:
  - `src/gpu_simulate_test/paper_fidelity/matrix.py`
  - `src/gpu_simulate_test/paper_fidelity/matrix_manifest.py`
  - `src/gpu_simulate_test/paper_fidelity/paper_models.py` if no longer used elsewhere

Optional (migration-friendly) alternative:
- Keep `matrix` as a stub for one cycle that exits non-zero with an explicit message:
  - “`paper-fidelity matrix` removed; use `scripts/paper_fidelity_sweep.sh` + `paper-fidelity aggregate`.”

### Step 3: Update docs/specs to remove “matrix” references and teach the new flow

Update or replace:
- `docs/runbooks/paper_fidelity_matrix.md` → `docs/runbooks/paper_fidelity_sweep.md` (omit aggregation for now)
- `docs/tutorial/howto/tut-paper-fidelity-static-and-dynamic.md` (remove matrix mentions; add sweep/aggregate examples)
- `specs/003-paper-fidelity-more-models/quickstart.md` (switch to `.sh` sweep; keep aggregation as “future”)
- `README.md` (entrypoint guidance)

### Step 4: Validation checklist (manual/CLI)

- `pixi run paper-fidelity --help` no longer shows `matrix`.
- Run one small-scale single-case sweep (1 model × static) and confirm:
  - report dir created under `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/...`
- Run a sweep with an intentional failure (e.g., missing model assets) and confirm:
  - `failure_record.json` is written for the failed case

## Impact Analysis

- **Breaking change**: any existing automation using `pixi run paper-fidelity matrix ...` will fail.
  - Mitigation: provide `scripts/paper_fidelity_sweep.sh` with comparable flags, and optionally keep a stub `matrix` subcommand that prints the migration path.
- **Behavior differences**:
  - The old `matrix` runner owned output structure (`paper_models_matrix_<run_id>`). The new flow should keep a similarly stable “sweep root” directory to preserve discoverability.
- **Deferred capability**: no combined/aggregated report is produced in this round.
  - Mitigation: keep a stable sweep log (`cases.jsonl`) and a stable output directory layout so aggregation can be added later without rerunning GPU work.
- **Risk: partial runs / interruptions**:
  - `.sh` sweeps can be interrupted mid-way, leaving mixed outputs.
  - Mitigation: write `cases.jsonl` incrementally (append per completed attempt) so aggregation can operate on partial state.

## Expected Outcome

- `paper-fidelity` CLI focuses on single-run operations (`trace`, `profile`, `repro`, `score`, `report`).
- Sweeps become transparent, editable, and override-friendly via `scripts/paper_fidelity_sweep.sh`.
- A future aggregation step can summarize sweep runs without requiring the expensive GPU work to be rerun.

## TODO

- [ ] Add `scripts/paper_fidelity_sweep.sh` (flags, looping, robust logging, emits `cases.jsonl`)
- [ ] Remove `paper-fidelity matrix` argparse wiring from `src/gpu_simulate_test/cli/paper_fidelity.py`
- [ ] Delete/retire `src/gpu_simulate_test/paper_fidelity/matrix.py` and `src/gpu_simulate_test/paper_fidelity/matrix_manifest.py`
- [ ] Delete/retire `src/gpu_simulate_test/paper_fidelity/paper_models.py` (or keep if reused by `aggregate`)
- [ ] Update docs: replace matrix runbook and adjust tutorials/quickstarts/README
- [ ] Run manual validation (1 success + 1 failure case) and confirm combined report correctness

## Example Refactor Snippets

### 1) CLI: remove `matrix` (no aggregation yet)

Before (`src/gpu_simulate_test/cli/paper_fidelity.py`, conceptual):

```py
matrix = sub.add_parser("matrix")
matrix.add_argument("--scale", choices=["small", "medium", "full"], default="small")
...
elif args.cmd == "matrix":
    if hydra_overrides:
        raise ValueError("matrix does not accept Hydra overrides")
    ...
    run_matrix(...)
```

After (conceptual):

```py
# no matrix parser
# sweeps move to `scripts/paper_fidelity_sweep.sh`
```

### 2) Sweep script: explicit per-case commands

Before: “one command does everything”:

```bash
pixi run paper-fidelity matrix --scale small --workloads static,dynamic --include-cpu-overhead --run-id X
```

After: `.sh` sweep explicitly runs each case:

```bash
TP=1
PP=1

profiling_root="$(
  pixi run paper-fidelity profile --scenario llama2_70b_arxiv --include-cpu-overhead \
    "scenario.real.parallel.tensor_parallel_size=${TP}" \
    "scenario.real.parallel.pipeline_parallel_size=${PP}" \
    "scenario.vidur.tensor_parallel_size=${TP}" \
    "scenario.vidur.num_pipeline_stages=${PP}" \
    "profiling.tensor_parallel_size=${TP}" \
    "profiling.num_gpus=$((TP * PP))" \
  | tail -n 1
)"
pixi run paper-fidelity repro --scenario llama2_70b_arxiv --workload static --scale small \
  "scenario.vidur.profiling_root=${profiling_root}" \
  "scenario.real.parallel.tensor_parallel_size=${TP}" \
  "scenario.real.parallel.pipeline_parallel_size=${PP}" \
  "scenario.vidur.tensor_parallel_size=${TP}" \
  "scenario.vidur.num_pipeline_stages=${PP}"
pixi run paper-fidelity repro --scenario llama2_70b_arxiv --workload dynamic --scale small \
  "scenario.vidur.profiling_root=${profiling_root}" \
  "scenario.real.parallel.tensor_parallel_size=${TP}" \
  "scenario.real.parallel.pipeline_parallel_size=${PP}" \
  "scenario.vidur.tensor_parallel_size=${TP}" \
  "scenario.vidur.num_pipeline_stages=${PP}"
```

## References

- Planning guideline: `magic-context/instructions/planning/make-refactor-plan.md`
- Current CLI entrypoint: `src/gpu_simulate_test/cli/paper_fidelity.py`
- Current matrix implementation: `src/gpu_simulate_test/paper_fidelity/matrix.py`
- Current matrix manifest schema: `src/gpu_simulate_test/paper_fidelity/matrix_manifest.py`
- Current matrix runbook: `docs/runbooks/paper_fidelity_matrix.md`
- Tutorial: `docs/tutorial/howto/tut-paper-fidelity-static-and-dynamic.md`
- Spec quickstart: `specs/003-paper-fidelity-more-models/quickstart.md`

- Context7 library ids (3rd-party):
  - Hydra: `/facebookresearch/hydra`
  - OmegaConf: `/omry/omegaconf`
