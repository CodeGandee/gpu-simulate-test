# Run workflow (workload → real → sim → report)

The workflow is split into five commands. Each command is a Hydra app, so you can override config values with `key=value` arguments.

All outputs go under `tmp/` by default.

## 1) Generate a deterministic workload

```bash
pixi run workload-spec model=qwen3_0_6b
```

Outputs:

- `tmp/workloads/<workload_id>/prompts.jsonl`
- `tmp/workloads/<workload_id>/trace_lengths.csv`
- `tmp/workloads/<workload_id>/trace_intervals.csv`
- `tmp/workloads/<workload_id>/workload_meta.json`

## 2) Run real timing (choose a backend)

### Option A: Transformers backend

```bash
pixi run real-bench \
  backend=transformers \
  workload.workload_dir=tmp/workloads/<workload_id>
```

### Option B: Sarathi-Serve backend

```bash
CUDA_VISIBLE_DEVICES=0 pixi run real-bench \
  backend=sarathi \
  model.model_id=$(pwd)/models/qwen3-0.6b/source-data \
  workload.workload_dir=tmp/workloads/<workload_id>
```

Outputs:

- `tmp/real_runs/<run_id>/request_metrics.csv`
- `tmp/real_runs/<run_id>/token_metrics.csv`
- `tmp/real_runs/<run_id>/run_meta.json`

## 3) Generate Vidur profiling bundle (one-time per model + hardware)

```bash
pixi run vidur-profile \
  model=qwen3_0_6b \
  hardware=a100 \
  vidur.profiling.root=tmp/vidur_profiling/a100/qwen3_0_6b
```

## 4) Run Vidur simulation

```bash
pixi run vidur-sim \
  model=qwen3_0_6b \
  hardware=a100 \
  vidur.profiling.root=tmp/vidur_profiling/a100/qwen3_0_6b \
  workload.workload_dir=tmp/workloads/<workload_id>
```

Outputs:

- `tmp/vidur_runs/<run_id>/request_metrics.csv`
- `tmp/vidur_runs/<run_id>/token_metrics.csv`
- `tmp/vidur_runs/<run_id>/run_meta.json`

Note: `vidur-sim` is **CPU-side simulation** driven by the profiling bundle; it does not execute end-to-end GPU inference.

## 5) Compare one real run vs one sim run

```bash
pixi run compare-runs \
  real_run_dir=tmp/real_runs/<run_id> \
  sim_run_dir=tmp/vidur_runs/<run_id>
```

Outputs:

- `tmp/comparisons/<comparison_id>/summary.md`
- `tmp/comparisons/<comparison_id>/tables/*.csv`
- `tmp/comparisons/<comparison_id>/figs/*`

## More detail

- Exact commands (pinned to the feature): `specs/001-compare-vidur-real-timing/quickstart.md`
- Runbook: `context/runbooks/001-compare-vidur-real-timing-troubleshooting.md`

