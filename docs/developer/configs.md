# Hydra configs

All user-facing commands are Hydra apps with presets under:

- `configs/compare_vidur_real/` (compare Vidur vs real timing)
- `configs/paper_fidelity/` (paper fidelity reproduction)

## Stage configs

### Compare workflow (`configs/compare_vidur_real/`)

- Workload generation: `configs/compare_vidur_real/workload_spec.yaml`
- Real timing: `configs/compare_vidur_real/real_bench.yaml`
- Vidur profiling: `configs/compare_vidur_real/vidur_profile.yaml`
- Vidur sim: `configs/compare_vidur_real/vidur_sim.yaml`
- Comparison: `configs/compare_vidur_real/compare_runs.yaml`

Each stage sets `hydra.run.dir` so the outputs land under `tmp/.../<stable_id>/`.

## Config groups

### Compare workflow (`configs/compare_vidur_real/`)

- `configs/compare_vidur_real/model/` (e.g. `qwen3_0_6b.yaml`)
- `configs/compare_vidur_real/hardware/` (e.g. `a100.yaml`)
- `configs/compare_vidur_real/workload/` (e.g. `default.yaml`)
- `configs/compare_vidur_real/backend/` (`transformers.yaml`, `sarathi.yaml`)
- `configs/compare_vidur_real/vidur/` (profiling root + model key)

### Paper fidelity workflow (`configs/paper_fidelity/`)

- Stage configs:
  - `configs/paper_fidelity/repro.yaml`
  - `configs/paper_fidelity/trace.yaml`
  - `configs/paper_fidelity/profile.yaml`
  - `configs/paper_fidelity/score.yaml`
- Groups:
  - `configs/paper_fidelity/scenario/` (model + trace source + Vidur/Sarathi knobs)
  - `configs/paper_fidelity/workload/` (`static` vs `dynamic` arrivals)

## Adding a new paper-fidelity scenario

Paper-fidelity “scenarios” are Hydra configs under `configs/paper_fidelity/scenario/`. The `--scenario <name>` flag selects one of these files by basename.

### 1) Create the scenario YAML

- Copy an existing scenario:

```bash
cp configs/paper_fidelity/scenario/llama2_7b_arxiv.yaml \
  configs/paper_fidelity/scenario/<new_scenario>.yaml
```

- Edit `configs/paper_fidelity/scenario/<new_scenario>.yaml`:
  - `name`: default artifact namespace (used in `tmp/paper_fidelity/...` and report naming).
  - `model.model_id`: the model identifier passed to Vidur (must be supported by Vidur / our Vidur integration).
  - `model.model_ref`: local path for Sarathi replay (typically `${paths.repo_root}/models/<model>/source-data`).
  - `trace_source`: where token lengths (and optionally arrivals) come from.
  - `vidur.*` and `real.*`: parity-critical knobs (scheduler + TP/PP + chunk/batch caps).
  - `capacity_search`: required for `--workload dynamic` (capacity discovery drives the operating QPS).

Trace sources (`scenario.trace_source.kind`):

- `vidur_processed_lengths_csv`: *untimed* lengths CSV → workflow generates `arrived_at`:
  - static: all `arrived_at=0`
  - dynamic: Poisson arrivals from `workload.qps` / capacity search
- `trace_csv`: already-timed canonical trace (`arrived_at` present).
  - static forces `arrived_at=0`; dynamic uses the arrivals in the CSV.
- `legacy_workload_dir`: load from a `tmp/workloads/<id>/...` style directory (timed).

### 2) Ensure model assets exist for Sarathi replay

Sarathi requires `scenario.model.model_ref` to exist on disk (weights/tokenizer). This repo typically uses the symlink pattern under `models/`.

- Create/repair the symlink:
  - `bash models/<model>/bootstrap.sh`
- Or add the model to `models/bootstrap.yaml` and run:
  - `bash models/bootstrap.sh`

### 3) Run host-matched profiling, then repro

For meaningful sim-vs-real % error reproduction, generate a host profiling root (microbenchmarks) and pass it to `repro`:

```bash
pixi run paper-fidelity profile --scenario <new_scenario> --include-cpu-overhead

pixi run paper-fidelity repro --scenario <new_scenario> --workload static \
  scenario.vidur.profiling_root=/abs/path/to/tmp/paper_fidelity/profiling_roots/<new_scenario>/<timestamp-dir>

pixi run paper-fidelity repro --scenario <new_scenario> --workload dynamic --scale medium \
  scenario.vidur.profiling_root=/abs/path/to/tmp/paper_fidelity/profiling_roots/<new_scenario>/<timestamp-dir>
```

## Common overrides

```bash
# Change prompts input + decode length
pixi run workload-spec \
  workload.prompts=tmp/prompts/example.prompts.jsonl \
  workload.num_decode_tokens=64 \
  workload.seed=123

# Run real-bench with Sarathi and load weights from local path
CUDA_VISIBLE_DEVICES=0 pixi run real-bench \
  backend=sarathi \
  model.model_id=$(pwd)/models/qwen3-0.6b/source-data \
  workload.workload_dir=tmp/workloads/<workload_id>

# Point Vidur to an explicit profiling root
pixi run vidur-sim \
  vidur.profiling.root=tmp/vidur_profiling/a100/qwen3_0_6b \
  workload.workload_dir=tmp/workloads/<workload_id>

# MLP profiling method selection is required (no hidden defaults).
pixi run vidur-profile \
  vidur.profiling.root=tmp/vidur_profiling/a100/qwen3_0_6b \
  profiling.mlp.profile_method=cuda_event

# Best-effort consumption of legacy profiling roots with missing (NaN) MLP timings:
pixi run vidur-sim \
  vidur.profiling.root=tmp/vidur_profiling/a100/qwen3_0_6b \
  workload.workload_dir=tmp/workloads/<workload_id> \
  vidur.validation.mlp.nan_policy=drop
```

See `docs/manual/mlp-validation-and-fallback.md` for detailed behavior of `mode`, `nan_policy`, and `profiling.mlp.fallback.*` combinations.

Paper fidelity overrides:

```bash
# Run a small trace subset (fast iteration)
pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic \
  trace_subset.kind=range trace_subset.begin=0 trace_subset.end=32

# Namespace runs explicitly (keep multiple runs side-by-side)
pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload static \
  scenario.name=llama2_7b_arxiv_sim_vs_real_2026-01-09_00-00-00_static_small

# Override profiling root (host-matched; recommended for meaningful sim-vs-real % error)
pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload static \
  scenario.vidur.profiling_root=/abs/path/to/tmp/paper_fidelity/profiling_roots/...

# Parity-critical knobs: do not rely on Vidur defaults (align with real runner unless studying drift)
pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload static \
  scenario.vidur.scheduler.type=sarathi \
  scenario.vidur.scheduler.chunk_size=16 \
  scenario.vidur.scheduler.batch_size_cap=16 \
  scenario.vidur.skip_cpu_overhead_modeling=false

# Host profiling includes CPU overhead microbenchmarks by default.
# Disable only for debugging:
#   pixi run paper-fidelity profile --scenario llama2_7b_arxiv --no-include-cpu-overhead
pixi run paper-fidelity profile --scenario llama2_7b_arxiv
```
