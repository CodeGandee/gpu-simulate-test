# Runbook: Reading paper-fidelity matrix outputs

This runbook explains how to interpret the outputs from:

```bash
pixi run paper-fidelity matrix --scale small --workloads static,dynamic --include-cpu-overhead
```

## Where outputs live

Each matrix run writes a dedicated directory under:

- `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/paper_models_matrix_<run_id>/`
  - `manifest.json`
  - `failures/*.json`

Each successful `paper-fidelity repro` run writes a report bundle under:

- `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<scenario_name_or_tag>/`
  - `summary.md`
  - `run_meta.json`
  - `scores.json`
  - `inputs/` (snapshotted inputs for portability)

If a single-scenario `paper-fidelity repro` fails, it writes:

- `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<scenario_name_or_tag>/failure_record.json`

## How to read `manifest.json`

`manifest.json` is a single JSON object that summarizes all attempted runs.

Key fields:

- `run_id`: matrix run identifier
- `scenarios`: scenario keys requested
- `workloads`: workloads requested (`static`, `dynamic`)
- `scale`: requested scale (`small`, `medium`, `full`)
- `runs`: list of per-scenario per-workload results

Each entry under `runs[*]` includes:

- `scenario_key`, `workload`, `scale`
- `status`: `success` or `failure`
- `report_dir` (on success)
- `failure_record_json` and `blocker_category` (on failure)

Quick triage helpers:

```bash
# List failed entries (requires jq)
jq -r '.runs[] | select(.status=="failure") | [.scenario_key,.workload,.scale,.blocker_category,.failure_record_json] | @tsv' \
  results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/paper_models_matrix_<run_id>/manifest.json
```

## How to read a failure record

Failure records are JSON objects with schema `v1`. Important fields:

- `action`: `trace|profile|repro|matrix`
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
- Reduce parallelism overrides for the scenario:
  - `scenario.real.parallel.tensor_parallel_size=<n>`
  - `scenario.real.parallel.pipeline_parallel_size=<n>`

### `missing model files`

Meaning: `scenario.model.model_ref` does not exist on this machine.

Remediation:

- Run the model bootstrap script and re-check `models/<model>/source-data`.

### `OOM`

Meaning: the run hit an out-of-memory condition during profiling or replay.

Remediation (typical options):

- Reduce model parallelism / concurrency (e.g., `scenario.real.scheduler.max_num_seqs`).
- Use a smaller scale for iteration (`--scale small`).

### `unsupported model`

Meaning: the real backend cannot load the chosen model architecture/config.

Remediation:

- Verify `scenario.model.model_id` and the model assets are compatible with the Sarathi backend in this repo.

