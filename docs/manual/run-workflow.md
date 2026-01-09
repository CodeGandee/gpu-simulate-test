# Run workflows

This repo has two user-facing workflows. All commands are Hydra apps, so you can override config values with `key=value` arguments.

All heavy outputs go under `tmp/` by default.

## A) Compare Vidur vs real timing (workload → real → sim → report)

This workflow is split into five commands.

### 1) Generate a deterministic workload

```bash
pixi run workload-spec model=qwen3_0_6b
```

Outputs:

- `tmp/workloads/<workload_id>/prompts.jsonl`
- `tmp/workloads/<workload_id>/trace_lengths.csv`
- `tmp/workloads/<workload_id>/trace_intervals.csv`
- `tmp/workloads/<workload_id>/workload_meta.json`

### 2) Run real timing (choose a backend)

#### Option A: Transformers backend

```bash
pixi run real-bench \
  backend=transformers \
  workload.workload_dir=tmp/workloads/<workload_id>
```

#### Option B: Sarathi-Serve backend

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

### 3) Generate Vidur profiling bundle (one-time per model + hardware)

```bash
pixi run vidur-profile \
  model=qwen3_0_6b \
  hardware=a100 \
  vidur.profiling.root=tmp/vidur_profiling/a100/qwen3_0_6b
```

### 4) Run Vidur simulation

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

### 5) Compare one real run vs one sim run

```bash
pixi run compare-runs \
  real_run_dir=tmp/real_runs/<run_id> \
  sim_run_dir=tmp/vidur_runs/<run_id>
```

Outputs:

- `tmp/comparisons/<comparison_id>/summary.md`
- `tmp/comparisons/<comparison_id>/tables/*.csv`
- `tmp/comparisons/<comparison_id>/figs/*`

## B) Paper fidelity reproduction (trace → sim + real → score → report)

This workflow is implemented as a single CLI with subcommands.

Notes:

- Treat `scenario.name` as the artifact namespace. Override it to keep multiple runs side-by-side (e.g. stamped run ids).
- For long runs, use `tmux` and write a log alongside the run outputs (e.g. `... 2>&1 | tee run.log`). See `context/instructions/run-lengthy-experiment.md`.

### 1) Generate a canonical trace (optional)

`paper-fidelity repro` generates a trace automatically, but `paper-fidelity trace` is useful for debugging trace generation/validation.

```bash
pixi run paper-fidelity trace --scenario llama2_7b_arxiv --workload static
pixi run paper-fidelity trace --scenario llama2_7b_arxiv --workload dynamic
```

Optional: run only a subset of requests (fast iteration):

```bash
# First 32 requests (range subset)
pixi run paper-fidelity trace --scenario llama2_7b_arxiv --workload dynamic \
  trace_subset.kind=range trace_subset.begin=0 trace_subset.end=32
```

### 2) End-to-end reproduction (recommended)

```bash
pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload static
pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic
```

Outputs:

- Trace: `tmp/paper_fidelity/traces/<scenario.name>/trace.csv`
- Sim metrics: `tmp/paper_fidelity/runs/<scenario.name>/sim/request_metrics.csv`
- Real metrics: `tmp/paper_fidelity/runs/<scenario.name>/real/request_metrics.csv`
- Report: `results/reports/<date>/paper_fidelity/<scenario.name>/summary.md`

### 3) Host-calibrated profiling (optional)

By default, scenarios point Vidur at the paper-provided profiling bundle under `extern/tracked/vidur`. To generate a host profiling root and rerun simulation with host-matched profiling:

```bash
pixi run paper-fidelity profile --scenario llama2_7b_arxiv
pixi run paper-fidelity profile --scenario llama2_7b_arxiv --include-cpu-overhead
pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload static \
  scenario.vidur.profiling_root=/abs/path/to/tmp/paper_fidelity/profiling_roots/...
```

Sim-vs-real parity note: do not rely on Vidur defaults. Set parity-critical knobs explicitly (scheduler type, chunk size, batch caps, CPU overhead modeling) and keep profiling, sim, and real runs aligned.

### 4) Score-only report (optional)

```bash
pixi run paper-fidelity score \
  --sim /abs/path/to/sim/request_metrics.csv \
  --real /abs/path/to/real/request_metrics.csv
```

## More detail

- Exact commands (pinned to the feature): `specs/001-compare-vidur-real-timing/quickstart.md`
- Runbook: `context/runbooks/001-compare-vidur-real-timing-troubleshooting.md`
- Paper fidelity quickstart: `specs/002-reproduce-vidur-paper-fidelity/quickstart.md`
