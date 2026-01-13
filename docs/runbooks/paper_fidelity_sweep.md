# Runbook: Reading paper-fidelity sweep outputs

This runbook explains how to interpret the outputs from the sweep script:

```bash
bash scripts/paper_fidelity_sweep.sh --scale small --workloads static,dynamic --tp 1 --pp 1 --run-id my_run_001
```

## Where outputs live

The sweep script writes:

- Sweep log:
  - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/sweep_<run_id>/cases.jsonl`
- Per-case report bundles (success):
  - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<scenario_tag>/`
    - `summary.md`
    - `run_meta.json`
    - `scores.json`
    - `inputs/` (snapshotted inputs for portability)
- Per-case failure records (failure):
  - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<scenario_tag>/failure_record.json` (repro failures)
  - `tmp/paper_fidelity/profiling_roots/<scenario_tag>/<timestamp-dir>/failure_record.json` (profile failures)

Notes:
- The sweep script overrides `scenario.name` to include the run id (e.g. `<scenario_key>_sweep_<run_id>`), so multiple sweeps on the same day do not collide.
- Dynamic reports use the CLI’s naming convention: `<scenario_tag>_dynamic_<scale>`.

## How to read `cases.jsonl`

`cases.jsonl` is append-only; each line is a JSON object describing one attempted action:

- `action`: `profile` or `repro`
- `scenario_key`: scenario config key under `configs/paper_fidelity/scenario/`
- `scenario_name`: the overridden `scenario.name` used for artifacts (`<scenario_key>_sweep_<run_id>`)
- `workload`: `static|dynamic` (only for `action=repro`)
- `scale`: `small|medium|full`
- `tp`, `pp`: global parallelism parameters used for the run
- `status`: `success|failure`
- `profiling_root` (profile success), `report_dir` (repro success)
- `output` (failure): last stdout line (often a `failure_record.json` path printed by the CLI)
- `error` (failure): last stderr line

Quick triage helpers:

```bash
# Show failures (requires jq)
jq -c 'select(.status=="failure")' results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/sweep_<run_id>/cases.jsonl

# Show per-case report dirs (requires jq)
jq -r 'select(.action=="repro" and .status=="success") | .report_dir' \
  results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/sweep_<run_id>/cases.jsonl
```

## How to read a failure record

Failure records are JSON objects with schema `v1`. Important fields:

- `action`: `trace|profile|repro`
- `scenario_key` / `scenario_name`
- `workload` / `scale` (nullable for profiling)
- `attempted_command`: argv list for the failing command (when available)
- `error_message` / `traceback`
- `blocker_category`: one of
  - `insufficient GPUs`
  - `OOM`
  - `missing model files`
  - `unsupported model`
  - `unknown`

## Common blockers and what to do

### `insufficient GPUs`

Meaning: the configured parallelism requires more visible GPUs than are available.

Remediation:

- Ensure `GSIM_CUDA_VISIBLE_DEVICES` is set (repo `.env` or exported).
- Run the sweep with smaller `--tp/--pp` (e.g., `--tp 1 --pp 1`).

### `missing model files`

Meaning: `scenario.model.model_ref` does not exist on this machine.

Remediation:

- Run the model bootstrap script and re-check `models/<model>/source-data`.

### `OOM`

Meaning: the run hit an out-of-memory condition during profiling or replay.

Remediation (typical options):

- Reduce `--tp/--pp`, or lower concurrency via overrides (e.g., `scenario.real.scheduler.max_num_seqs`).
- Use a smaller `--scale` for iteration.

### `unsupported model`

Meaning: the real backend cannot load the chosen model architecture/config.

Remediation:

- Verify `scenario.model.model_id` and the model assets are compatible with the Sarathi backend in this repo.

