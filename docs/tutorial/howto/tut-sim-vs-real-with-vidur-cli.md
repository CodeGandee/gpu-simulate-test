# How to produce sim-vs-real static and dynamic reports with `vidur-cli`

## Question
How do I produce **static** and **dynamic** sim-vs-real reports (Vidur sim vs real replay) using this repo’s `vidur-cli` workflow?

## Prerequisites

- **Environment:** `pixi` installed and `pixi install` has been run.
- **GPU:** A working CUDA setup (`pixi run python -c "import torch; print(torch.cuda.is_available())"` prints `True`).
- **GPU pinning:** Before running GPU stages (`svr profile/sim/real`), pin GPUs via `GSIM_CUDA_VISIBLE_DEVICES` (repo-local `.env` can set it; see `context/instructions/prep-dev-env.md`).
  - On this machine, use: `export GSIM_CUDA_VISIBLE_DEVICES=4,5` (optionally also `export CUDA_VISIBLE_DEVICES=4,5`).
- **Submodules:** `git submodule update --init --recursive` has been run (needed for Vidur + Sarathi).
- **Model assets:** The selected model preset’s `model.tokenizer_ref` exists.
  - For the example model in this tutorial (`llama2_7b`): `bash models/llama2-7b-hf/bootstrap.sh`.
- **Real backend import works (Sarathi backend):** `pixi run python -c "import sarathi; print('ok')"` succeeds.
- **Host-matched profiling:** Run `vidur-cli svr profile` on this machine before `vidur-cli svr sim` (this is required for meaningful sim-vs-real comparisons).

## Implementation Idea

`vidur-cli` is a **resumable pipeline** anchored on a single `<run_dir>`:

1. Create a run directory with selected presets (`svr init-run`).
2. Materialize a canonical token-length trace (`svr trace`).
3. Run profiling (`svr profile`) and record the profiling root in `run_state.json`.
4. Run Vidur simulation (`svr sim`) using the trace + profiling root.
5. Run real replay (`svr real`) using the same trace.
6. Generate a comparison report (`svr report`) under `<run_dir>/report/summary.md`.

Static vs dynamic is controlled by the **arrival schedule** used when generating the trace:

- **Static:** `workload.arrival.kind=fixed_interval` with `workload.arrival.inter_arrival_ns=0` (all arrivals at time 0).
- **Dynamic:** `workload.arrival.kind=poisson` with `workload.arrival.poisson_rate_per_s=<qps>` (timed arrivals).

## Step-by-Step with Code

### Step 1: Confirm environment and assets

```bash
git submodule update --init --recursive
pixi install

# Optional but recommended: ensure GSIM_CUDA_VISIBLE_DEVICES is set (repo-local).
test -f .env && sed -n '1,50p' .env || true

# If you're on this machine, pin to the available GPUs:
export GSIM_CUDA_VISIBLE_DEVICES=4,5
export CUDA_VISIBLE_DEVICES=4,5

pixi run python -c "import torch; print('cuda_available=', torch.cuda.is_available())"

# Ensure model/tokenizer reference exists for the example model preset.
bash models/llama2-7b-hf/bootstrap.sh
ls -la models/llama2-7b-hf/source-data
```

### Step 2: Pick `<pwd>` and a workspace directory (recommended)

By default, `vidur-cli` uses a workspace under `<pwd>/.vidur-output/<workspace>/...`.

For most experiments, it’s more convenient to explicitly put the workspace under:

- `<pwd>/tmp/<experiment-name>/` (recommended; easy to clean up; if `<pwd>` is the repo root, `tmp/` is git-ignored)

```bash
# Run from the repo root (so <pwd> is the repo root):
export GSIM_REPO_ROOT="$PWD"

# Put all `vidur-cli` run directories under <pwd>/tmp/<experiment-name>/...
EXP_NAME="vidur-cli-sim-vs-real"
export GSIM_VIDUR_WORKSPACE_DIR="$PWD/tmp/$EXP_NAME"
mkdir -p "$GSIM_VIDUR_WORKSPACE_DIR"
```

### Step 3: Preflight (resources + available presets)

```bash
pixi run -m "$GSIM_REPO_ROOT" vidur-cli resources show
pixi run -m "$GSIM_REPO_ROOT" vidur-cli configs list --group model
pixi run -m "$GSIM_REPO_ROOT" vidur-cli configs list --group backend
pixi run -m "$GSIM_REPO_ROOT" vidur-cli configs list --group workload
```

### Step 4: Produce the static report

This produces a run where all requests arrive at time 0 (static).

#### 4.1 Create a run directory

```bash
RUN_DIR=$(
  pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr init-run \
    model=llama2_7b hardware=a100 backend=sarathi workload=default vidur=default
)
echo "RUN_DIR=$RUN_DIR"
```

Note: `backend=transformers` is useful for a quick smoke test, but it will not match Vidur’s batching/scheduling behavior (so sim-vs-real comparisons can drift a lot). For paper-fidelity-like comparisons, use `backend=sarathi`.

For `backend=sarathi`, the replay/scheduler parity knobs live in `configs/compare_vidur_real/backend/sarathi.yaml` (defaults: `chunk_size=16`, `max_num_seqs=16`).

#### 4.2 Create the canonical trace

Option A (recommended, paper-fidelity-like): use the Arxiv processed lengths for LLaMA2, and subset to a small file:

```bash
ARXIV_LENGTHS_CSV="$GSIM_REPO_ROOT/extern/tracked/vidur/data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv"
INPUTS_DIR="$GSIM_VIDUR_WORKSPACE_DIR/inputs"
mkdir -p "$INPUTS_DIR"
head -n 51 "$ARXIV_LENGTHS_CSV" > "$INPUTS_DIR/lengths_arxiv_small.csv"

pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr trace --run-dir "$RUN_DIR" --from-lengths "$INPUTS_DIR/lengths_arxiv_small.csv"
```

Option B (quick smoke): provide a tiny hand-written lengths-only CSV:

```bash
INPUTS_DIR="$GSIM_VIDUR_WORKSPACE_DIR/inputs"
mkdir -p "$INPUTS_DIR"
cat > "$INPUTS_DIR/lengths.csv" <<'EOF'
num_prefill_tokens,num_decode_tokens
256,64
512,64
1024,64
EOF

pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr trace --run-dir "$RUN_DIR" --from-lengths "$INPUTS_DIR/lengths.csv"
```

#### 4.3 Run profiling → sim → real → report

```bash
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr profile --run-dir "$RUN_DIR"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr sim     --run-dir "$RUN_DIR"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr real    --run-dir "$RUN_DIR"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr report  --run-dir "$RUN_DIR"
```

Expected output:

- Each stage prints its primary output path (e.g., `$RUN_DIR/profile`, `$RUN_DIR/sim`, `$RUN_DIR/real`, `$RUN_DIR/report/summary.md`).
- The final report is: `$RUN_DIR/report/summary.md`.

### Step 5: Produce the dynamic report (Poisson arrivals)

Unlike the `paper-fidelity` workflow, `vidur-cli` does **not** do an automatic capacity search; you must choose the QPS (`workload.arrival.poisson_rate_per_s`) yourself.

#### 5.1 Create a new run directory (dynamic arrival overrides)

```bash
RUN_DIR_DYN=$(
  pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr init-run \
    model=llama2_7b hardware=a100 backend=sarathi workload=default vidur=default \
    workload.arrival.kind=poisson \
    workload.arrival.poisson_rate_per_s=0.85 \
    workload.arrival.seed=42
)
echo "RUN_DIR_DYN=$RUN_DIR_DYN"
```

#### 5.2 Create the dynamic trace and run stages

```bash
# If you used the quick-smoke `lengths.csv` in Step 4.2 (Option B), swap this path to `$INPUTS_DIR/lengths.csv`.
INPUTS_DIR="$GSIM_VIDUR_WORKSPACE_DIR/inputs"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr trace --run-dir "$RUN_DIR_DYN" --from-lengths "$INPUTS_DIR/lengths_arxiv_small.csv"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr profile --run-dir "$RUN_DIR_DYN"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr sim     --run-dir "$RUN_DIR_DYN"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr real    --run-dir "$RUN_DIR_DYN"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr report  --run-dir "$RUN_DIR_DYN"
```

## Where outputs are written

Within each run directory:

- `run_state.json` (stage state + artifact pointers)
- `resources.json` (resolved resources + provenance)
- `resolved_config.yaml` (best-effort snapshot for provenance)
- `trace/trace.csv`, `trace/trace_meta.json`, plus compatibility CSVs (`trace_lengths.csv`, `trace_intervals.csv`)
- `profile/`, `sim/`, `real/`
- `report/summary.md` (final report) plus optional `report/tables/*`, `report/figs/*`

## How the timed trace is generated (dynamic) and where to find it

`svr trace` writes:

- Canonical trace: `$RUN_DIR/trace/trace.csv` with (at minimum) columns:
  - `request_id`, `arrival_time_ns`, `num_prefill_tokens`, `num_decode_tokens`
- Metadata: `$RUN_DIR/trace/trace_meta.json` with:
  - `arrival_schedule.kind` and parameters (seed, poisson rate or fixed interval)

In **static**, `arrival_time_ns` will be all zeros (or non-decreasing zeros).
In **dynamic**, `arrival_time_ns` is generated from the configured Poisson process (monotonic, in nanoseconds).

## Common diagnostics

- Print resolved resources and Hydra config roots before any command: `vidur-cli --print-resolved ...`
- If a stage fails, inspect: `$RUN_DIR/failure.json` (stage + message + optional context)
- Inspect stage outputs and pointers in: `$RUN_DIR/run_state.json`

## Complete Runnable Script

```bash
#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root (so <pwd> is the repo root):
export GSIM_REPO_ROOT="$PWD"

# If you're on this machine, pin to the available GPUs:
export GSIM_CUDA_VISIBLE_DEVICES=4,5
export CUDA_VISIBLE_DEVICES=4,5

EXP_NAME="vidur-cli-sim-vs-real"
export GSIM_VIDUR_WORKSPACE_DIR="$PWD/tmp/$EXP_NAME"
mkdir -p "$GSIM_VIDUR_WORKSPACE_DIR"

ARXIV_LENGTHS_CSV="$GSIM_REPO_ROOT/extern/tracked/vidur/data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv"
INPUTS_DIR="$GSIM_VIDUR_WORKSPACE_DIR/inputs"
mkdir -p "$INPUTS_DIR"
head -n 51 "$ARXIV_LENGTHS_CSV" > "$INPUTS_DIR/lengths_arxiv_small.csv"

# Static
RUN_DIR=$(
  pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr init-run \
    model=llama2_7b hardware=a100 backend=sarathi workload=default vidur=default
)
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr trace   --run-dir "$RUN_DIR" --from-lengths "$INPUTS_DIR/lengths_arxiv_small.csv"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr profile --run-dir "$RUN_DIR"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr sim     --run-dir "$RUN_DIR"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr real    --run-dir "$RUN_DIR"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr report  --run-dir "$RUN_DIR"

# Dynamic (choose QPS)
RUN_DIR_DYN=$(
  pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr init-run \
    model=llama2_7b hardware=a100 backend=sarathi workload=default vidur=default \
    workload.arrival.kind=poisson workload.arrival.poisson_rate_per_s=0.85 workload.arrival.seed=42
)
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr trace   --run-dir "$RUN_DIR_DYN" --from-lengths "$INPUTS_DIR/lengths_arxiv_small.csv"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr profile --run-dir "$RUN_DIR_DYN"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr sim     --run-dir "$RUN_DIR_DYN"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr real    --run-dir "$RUN_DIR_DYN"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr report  --run-dir "$RUN_DIR_DYN"
```
