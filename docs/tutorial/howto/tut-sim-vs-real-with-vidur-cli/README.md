# How to produce sim-vs-real static and dynamic reports with `vidur-cli`

This directory is a **self-contained, git-tracked** tutorial + demo for producing Vidur **sim vs real replay** reports using this repo’s `vidur-cli` workflow.

It demonstrates an end-to-end run for **LLaMA2-7B** on **A100** using the **Sarathi** real backend, producing a final sim-vs-real report under `<run_dir>/report/summary.md`.

## Quickstart Demo (self-contained, tracked)

From the repo root:

```bash
docs/tutorial/howto/tut-sim-vs-real-with-vidur-cli/run_demo_static_from_pf_trace.sh
```

The script:

- pins GPUs via `GSIM_CUDA_VISIBLE_DEVICES` (defaults to `4,5`)
- creates a fresh workspace under `<repo>/tmp/`
- runs: `init-run → trace(import) → profile → sim → real → report`
- prints the final report path (`<run_dir>/report/summary.md`)

Tracked demo artifacts in this directory:

- `inputs/trace.csv`: paper-fidelity-style trace snapshot (`arrived_at` seconds). This was copied from:
  - `results/reports/2026-01-15/paper_fidelity/llama2_7b_arxiv_vc_profile_static_small_20260115/inputs/trace.csv`
- `inputs/trace_import.csv`: `vidur-cli` canonical trace import format (`arrival_time_ns` nanoseconds).
- `expected_report/`: a representative `<run_dir>/report/` directory snapshot from one successful run on this repo.
  - Exact numbers may vary across machines.
  - Machine-local paths are sanitized to placeholders like `<RUN_DIR>` and `<MODEL_REF>`.
  - The key goal is that the artifact structure matches and the report includes the “apple-to-apple” config section.

### Maintainers: refresh `expected_report/`

```bash
docs/tutorial/howto/tut-sim-vs-real-with-vidur-cli/run_demo_static_from_pf_trace.sh --snapshot-report
```

This writes a sanitized report snapshot under the workspace:

- `$GSIM_VIDUR_WORKSPACE_DIR/report_snapshot_<run_id>/`

To update the git-tracked `expected_report/`, copy the snapshot into:

- `docs/tutorial/howto/tut-sim-vs-real-with-vidur-cli/expected_report/`

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

#### 4.4 (Optional) Match a `paper-fidelity` run to the same profiling + trace (static)

If you want to compare `vidur-cli` against `paper-fidelity` **apples-to-apples**, run `paper-fidelity repro`
while explicitly pointing it at:

- the same profiling root produced by `vidur-cli` (`$RUN_DIR/profile`)
- the same lengths CSV used by `vidur-cli svr trace`

```bash
# Unique scenario name so you don't overwrite other tmp/paper_fidelity outputs.
PF_SCENARIO="llama2_7b_arxiv_match_vidur_cli_static_$(date -u +%Y%m%dT%H%M%SZ)"

pixi run paper-fidelity repro \
  --scenario llama2_7b_arxiv \
  --workload static \
  --scale small \
  "scenario.name=$PF_SCENARIO" \
  "scenario.trace_source.path=$INPUTS_DIR/lengths_arxiv_small.csv" \
  "scenario.vidur.profiling_root=$RUN_DIR/profile"
```

Expected output:
- A report under `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/$PF_SCENARIO/summary.md`
- The trace snapshot in that report (`inputs/trace.csv`) has the same 50 `(num_prefill_tokens, num_decode_tokens)` pairs as your `vidur-cli` run.

### Step 5: Produce the dynamic report (Poisson arrivals)

Unlike the `paper-fidelity` workflow, `vidur-cli` does **not** do an automatic capacity search; you must choose the QPS (`workload.arrival.poisson_rate_per_s`) yourself.

If you want a paper-fidelity-like “85% of capacity” operating point:

1. Run a paper-fidelity **dynamic** repro once (it writes `capacity.json` under `tmp/paper_fidelity/runs/<scenario>/capacity/`).
2. Read `qps_85` from that `capacity.json`.
3. Use that value as `workload.arrival.poisson_rate_per_s` for `vidur-cli`.

#### 5.1 Create a new run directory (dynamic arrival overrides)

```bash
# Choose a QPS (requests/second). For paper-fidelity-like runs, use the `qps_85` from capacity search.
QPS="0.85"

RUN_DIR_DYN=$(
  pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr init-run \
    model=llama2_7b hardware=a100 backend=sarathi workload=default vidur=default \
    workload.arrival.kind=poisson \
    workload.arrival.poisson_rate_per_s="$QPS" \
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

### (Optional) Match `paper-fidelity` dynamic arrivals exactly

For dynamic, matching just the QPS is not enough if you want *identical arrival times*; you must also match the generated arrival schedule.
`paper-fidelity` produces its timed trace as `inputs/trace.csv` in the report directory (schema uses `arrived_at` in seconds).

To run `vidur-cli` on the **same timed trace** *and* the **same profiling root**:

1) Create a fresh `vidur-cli` run and do `svr profile` once.  
2) Run `paper-fidelity repro --workload dynamic` while pointing it at that same profiling root (so sim uses identical profiling).  
3) Convert the paper-fidelity `inputs/trace.csv` into `vidur-cli`’s canonical trace format (`arrival_time_ns` in nanoseconds) and import it into the same `vidur-cli` run.  
4) Run `svr sim/real/report`.

```bash
# 1) Create a fresh vidur-cli run and profile once (profiling is trace-independent).
RUN_DIR_MATCH=$(
  pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr init-run \
    model=llama2_7b hardware=a100 backend=sarathi workload=default vidur=default
)
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr profile --run-dir "$RUN_DIR_MATCH"

# 2) Run paper-fidelity dynamic once to produce a timed trace snapshot (using the same profiling root).
PF_DYN_SCENARIO="llama2_7b_arxiv_match_vidur_cli_dynamic_$(date -u +%Y%m%dT%H%M%SZ)"
PF_REPORT_DIR=$(
  pixi run paper-fidelity repro \
    --scenario llama2_7b_arxiv \
    --workload dynamic \
    --scale small \
    "scenario.name=$PF_DYN_SCENARIO" \
    "scenario.trace_source.path=$INPUTS_DIR/lengths_arxiv_small.csv" \
    "scenario.vidur.profiling_root=$RUN_DIR_MATCH/profile" \
    2>&1 | grep -E '^/.*results/reports/' | tail -n 1
)

# 3) Convert paper-fidelity trace (arrived_at seconds) -> vidur-cli canonical trace (arrival_time_ns int),
#    then import it into the same vidur-cli run.
PF_TRACE_CSV="$PF_REPORT_DIR/inputs/trace.csv"
VC_IMPORT_TRACE="$INPUTS_DIR/trace_import_from_paper_fidelity.csv"
export PF_TRACE_CSV VC_IMPORT_TRACE
pixi run python - <<'PY'
from pathlib import Path
import os
import pandas as pd

src = Path(os.environ["PF_TRACE_CSV"]).expanduser().resolve()
out = Path(os.environ["VC_IMPORT_TRACE"]).expanduser().resolve()
df = pd.read_csv(src)
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    {
        "request_id": df["request_id"].astype("int64"),
        "arrival_time_ns": (df["arrived_at"].astype(float) * 1e9).round().astype("int64"),
        "num_prefill_tokens": df["num_prefill_tokens"].astype("int64"),
        "num_decode_tokens": df["num_decode_tokens"].astype("int64"),
    }
).to_csv(out, index=False)
print(out)
PY
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr trace --run-dir "$RUN_DIR_MATCH" --import-trace "$VC_IMPORT_TRACE"

# 4) Run vidur-cli sim/real/report on that exact timed trace.
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr sim    --run-dir "$RUN_DIR_MATCH"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr real   --run-dir "$RUN_DIR_MATCH"
pixi run -m "$GSIM_REPO_ROOT" vidur-cli svr report --run-dir "$RUN_DIR_MATCH"
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
