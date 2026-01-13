# How to implement a new paper-fidelity scenario with `paper-fidelity`

## Question
How do I implement a new paper-fidelity scenario (so I can run `pixi run paper-fidelity profile|repro --scenario <name>`) in this repo?

## Prerequisites

- **Environment:** `pixi install` has been run; commands are executed via `pixi run ...`.
- **Submodules:** `git submodule update --init --recursive` (Vidur + Sarathi integration).
- **GPU:** CUDA works on the target machine (profiling + Sarathi replay are GPU workflows).
- **Model assets:** You have a local model directory that Sarathi can load, referenced via `scenario.model.model_ref`.
- **Basic Hydra familiarity:** you can pass `key=value` overrides on the CLI.

## Implementation Idea

**Approach:**
1. Add a new Hydra config under `configs/paper_fidelity/scenario/<name>.yaml`.
2. Ensure the scenario defines:
   - model identity for Vidur (`model.model_id`) and model assets for Sarathi (`model.model_ref`)
   - a trace source (`trace_source.kind` + `trace_source.path`) and its token limits/seed
   - parity-critical settings for Vidur and Sarathi (scheduler + TP/PP + chunk/max seqs)
   - capacity discovery settings (needed for dynamic runs)
3. Generate a host-matched profiling root (`paper-fidelity profile`) and run `paper-fidelity repro` using it.

**Key distinction to keep in mind:**
- `--scenario <name>` selects a file in `configs/paper_fidelity/scenario/`.
- `scenario.name` (inside the YAML) is the artifact namespace used under `tmp/paper_fidelity/...` and for report naming.

## Step-by-Step with Code

### Step 1: Pick a scenario name and copy a working baseline

Start from a known-good scenario and edit only what you need.

```bash
cp configs/paper_fidelity/scenario/llama2_7b_arxiv.yaml \
  configs/paper_fidelity/scenario/<new_scenario>.yaml
```

### Step 2: Fill in the required scenario fields

Edit `configs/paper_fidelity/scenario/<new_scenario>.yaml`.

At minimum, ensure these blocks are correct:

#### `name`

This is the default namespace for outputs.

```yaml
name: <new_scenario>
```

#### `model` (Vidur id + Sarathi model assets)

```yaml
model:
  # Vidur model identifier (must be supported by Vidur / our integrations).
  model_id: meta-llama/Llama-2-7b-hf
  # Sarathi loads weights/tokenizer from this local path.
  model_ref: ${paths.repo_root}/models/<model>/source-data
```

If you don’t want to use the repo’s `models/<model>/source-data` symlink pattern, you can set `model_ref` to an absolute path.

#### `trace_source` (where the “canonical” request lengths come from)

Pick one `kind`:

1) **Untimed lengths CSV → workflow generates arrivals**:

```yaml
trace_source:
  kind: vidur_processed_lengths_csv
  path: ${paths.repo_root}/extern/tracked/vidur/data/processed_traces/<file>.csv
  max_tokens: 4096
  seed: 42
  num_requests: null
```

2) **Already-timed trace** (`arrived_at` present):

```yaml
trace_source:
  kind: trace_csv
  path: ${paths.repo_root}/path/to/trace.csv
  max_tokens: 4096
  seed: 42
  num_requests: null
```

3) **Legacy workload directory** (`tmp/workloads/<id>/...` style):

```yaml
trace_source:
  kind: legacy_workload_dir
  path: ${paths.repo_root}/tmp/workloads/<workload_id>
  max_tokens: 4096
  seed: 42
  num_requests: null
```

Notes:
- For `vidur_processed_lengths_csv`, the workflow can do `trace_subset.kind=indices=...` because it can “rebuild” arrivals.
- For timed sources (`trace_csv`, `legacy_workload_dir`), use `trace_subset.kind=range` (index slicing); indices are not supported.

#### `vidur` (sim side)

Ensure you set parity-critical knobs explicitly (don’t rely on Vidur defaults).

```yaml
vidur:
  profiling_root: ${paths.repo_root}/extern/tracked/vidur
  device: a100
  network_device: a100_pairwise_nvlink
  tensor_parallel_size: 1
  num_pipeline_stages: 1
  seed: 42

  # Typically overridden to a host profiling root at runtime:
  # scenario.vidur.profiling_root=/abs/path/to/tmp/paper_fidelity/profiling_roots/<scenario>/<timestamp>

  # CPU overhead modeling toggles:
  skip_cpu_overhead_modeling: true
  cpu_overhead:
    validation: strict

  scheduler:
    type: sarathi
    chunk_size: ${scenario.real.scheduler.chunk_size}
    batch_size_cap: ${scenario.real.scheduler.max_num_seqs}
    block_size: 16
    watermark_blocks_fraction: 0.01
```

#### `real` (Sarathi replay side)

```yaml
real:
  backend: sarathi
  sampling:
    ignore_eos: true
  scheduler:
    chunk_size: 16
    max_num_seqs: 16
  parallel:
    tensor_parallel_size: 1
    pipeline_parallel_size: 1
  metrics:
    write_metrics: true
    enable_chrome_trace: false
```

#### `capacity_search` (required for dynamic runs)

Dynamic paper-fidelity runs discover a capacity QPS, then run at `qps_85` by default.

```yaml
capacity_search:
  enabled: true
  min_qps: 1.0
  max_qps: 20.0
  max_iters: 6
  overload_p99_scheduling_delay_s: 5.0
  qps_operating_point_fraction: 0.85
```

### Step 3: Ensure Sarathi can load the model assets

If you’re using the repo’s `models/` symlink pattern, add the model under `models/<model>/` and create the `source-data` symlink.

Example:

```bash
bash models/<model>/bootstrap.sh
ls -la models/<model>/source-data
```

If you want the model included in `bash models/bootstrap.sh`, add it to `models/bootstrap.yaml`.

### Step 4: Generate a host-matched profiling root (required for meaningful % error)

```bash
pixi run paper-fidelity profile --scenario <new_scenario> --include-cpu-overhead
```

This prints a path like:
- `tmp/paper_fidelity/profiling_roots/<new_scenario>/<timestamp-dir>/`

Use that path as `scenario.vidur.profiling_root=...` for repro runs.

### Step 5: Run repro (static and dynamic)

```bash
PROFILING_ROOT="/abs/path/to/tmp/paper_fidelity/profiling_roots/<new_scenario>/<timestamp-dir>"

pixi run paper-fidelity repro --scenario <new_scenario> --workload static \
  "scenario.vidur.profiling_root=${PROFILING_ROOT}"

pixi run paper-fidelity repro --scenario <new_scenario> --workload dynamic --scale medium \
  "scenario.vidur.profiling_root=${PROFILING_ROOT}"
```

### Complete Runnable Script

```bash
#!/usr/bin/env bash
set -euo pipefail

scenario="<new_scenario>"

git submodule update --init --recursive
pixi install

# Optional if using `models/<model>/source-data` symlink pattern:
# bash models/<model>/bootstrap.sh

profiling_root="$(pixi run paper-fidelity profile --scenario "${scenario}" --include-cpu-overhead | tail -n 1)"

pixi run paper-fidelity repro --scenario "${scenario}" --workload static \
  "scenario.vidur.profiling_root=${profiling_root}"

pixi run paper-fidelity repro --scenario "${scenario}" --workload dynamic --scale medium \
  "scenario.vidur.profiling_root=${profiling_root}"
```

### [Optional] Alternative Interface (debug trace-only)

To validate trace generation without running Vidur/Sarathi:

```bash
pixi run paper-fidelity trace --scenario <new_scenario> --workload static
pixi run paper-fidelity trace --scenario <new_scenario> --workload dynamic --scale medium
```

Outputs:
- `tmp/paper_fidelity/traces/<scenario.name>/trace.csv`
- `tmp/paper_fidelity/traces/<scenario.name>/trace_meta.json`

## Input and Output

### Input

- `configs/paper_fidelity/scenario/<new_scenario>.yaml`: scenario definition.
- Model assets directory referenced by `scenario.model.model_ref`.
- Trace source referenced by `scenario.trace_source.path`.

### Output

- Host profiling root:
  - `tmp/paper_fidelity/profiling_roots/<scenario.name>/<timestamp-dir>/`
- Canonical trace:
  - `tmp/paper_fidelity/traces/<scenario.name>/trace.csv`
- Repro outputs (mutable, overwritten on reruns):
  - `tmp/paper_fidelity/runs/<scenario.name>/{sim,real}/request_metrics.csv`
  - `tmp/paper_fidelity/runs/<scenario.name>/capacity/capacity.json` (dynamic)
- Reports (stable per report dir; snapshots inputs under `inputs/`):
  - `results/reports/<UTC-YYYY-MM-DD>/paper_fidelity/<report_scenario>/summary.md`

## References

### Relevant Source Code

- `src/gpu_simulate_test/cli/paper_fidelity.py`: selects scenario configs and orchestrates `profile`, `trace`, `repro`.
- `configs/paper_fidelity/profile.yaml`: profiling defaults; uses `scenario.*` fields for TP/max tokens.
- `src/gpu_simulate_test/paper_fidelity/profiling.py`: host profiling implementation.
- `src/gpu_simulate_test/paper_fidelity/traces.py`: trace schema + Poisson arrivals + validators.
- `src/gpu_simulate_test/paper_fidelity/capacity.py`: dynamic capacity discovery.
- `src/gpu_simulate_test/vidur_ext/sim_runner.py`: Vidur paper-fidelity sim wrapper.
- `src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py`: Sarathi paper-fidelity replay.

### Online Resources

- Vidur (vendored source): `extern/tracked/vidur/`
- Sarathi-Serve (vendored source): `extern/tracked/sarathi-serve/`

