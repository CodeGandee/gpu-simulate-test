# Quickstart: Reproduce Vidur paper fidelity

**Spec**: `<WORKSPACE_ROOT>/specs/002-reproduce-vidur-paper-fidelity/spec.md`  
**Plan**: `<WORKSPACE_ROOT>/specs/002-reproduce-vidur-paper-fidelity/plan.md`  
**Date**: 2026-01-05

This quickstart describes the end-to-end workflow for the paper-fidelity reproduction feature.

**Path convention**: `<WORKSPACE_ROOT>` refers to the repository root.

## Prerequisites

1. Initialize dependencies:

   - `git submodule update --init --recursive`

2. Create/update the Pixi environment:

   - `pixi install`

3. Verify CUDA is available (required for real runs):

   - `pixi run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`

4. Ensure the baseline model reference is bootstrapped:

   - `bash <WORKSPACE_ROOT>/models/llama2-7b-hf/bootstrap.sh`

## Baseline scenario (MVP)

Default scenario: **LLaMA2-7B + arXiv summarization token-length trace**.

The baseline scenario config is expected at:

- `<WORKSPACE_ROOT>/configs/paper_fidelity/scenario/llama2_7b_arxiv.yaml`

## End-to-end reproduction

Run **static** fidelity (paper “offline” workload):

- `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload static`

Run **dynamic** fidelity (paper “online” workload at 85% capacity):

- `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic`

## Fast iteration (trace subset)

To run a small, deterministic subset of the trace (useful for quick debugging):

- Built-in scale presets:
  - Small (first 50 requests): `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic --scale small`
  - Medium (first 500 requests): `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic --scale medium`
  - Full (default): `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic --scale full`

- First 32 requests (range subset):
  - `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic trace_subset.kind=range trace_subset.begin=0 trace_subset.end=32`
- Discrete indices (only for untimed trace sources, e.g. `vidur_processed_lengths_csv`):
  - `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic trace_subset.kind=indices trace_subset.indices=[0,3,10,42]`

Expected artifacts:

- Trace: `<WORKSPACE_ROOT>/tmp/paper_fidelity/traces/llama2_7b_arxiv/trace.csv`
- Sim metrics: `<WORKSPACE_ROOT>/tmp/paper_fidelity/runs/llama2_7b_arxiv/sim/request_metrics.csv`
- Real metrics: `<WORKSPACE_ROOT>/tmp/paper_fidelity/runs/llama2_7b_arxiv/real/request_metrics.csv`
- Capacity (dynamic): `<WORKSPACE_ROOT>/tmp/paper_fidelity/runs/llama2_7b_arxiv/capacity/capacity.json`
- Report: `<WORKSPACE_ROOT>/results/reports/<date>/paper_fidelity/llama2_7b_arxiv/summary.md`

## Host-calibrated “gap reproduction” (optional)

By default, scenarios use Vidur’s **paper-provided profiling bundle** (`extern/tracked/vidur`) for simulation. This is useful for a pipeline sanity check, but the sim-vs-real gap may drift on a different host stack.

To make the sim-vs-real comparison meaningful on *this* host, first generate a **host profiling root**:

- `pixi run paper-fidelity profile --scenario llama2_7b_arxiv`

The command prints the profiling root path (under `<WORKSPACE_ROOT>/tmp/paper_fidelity/profiling_roots/...`). Then run repro while overriding the scenario’s profiling root:

- `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload static scenario.vidur.profiling_root=/abs/path/to/tmp/paper_fidelity/profiling_roots/...`
- `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic scenario.vidur.profiling_root=/abs/path/to/tmp/paper_fidelity/profiling_roots/...`

Additional artifacts:

- Profiling outputs (large, intermediate): `<WORKSPACE_ROOT>/tmp/paper_fidelity/profiling_outputs/<scenario>/<run_id>/...`
- Host profiling root: `<WORKSPACE_ROOT>/tmp/paper_fidelity/profiling_roots/<scenario>/<run_id>/data/profiling/...`
- Profiling provenance: `<WORKSPACE_ROOT>/tmp/paper_fidelity/profiling_roots/<scenario>/<run_id>/profiling_meta.json`

Reports include a `## Profiling` section indicating `mode=paper|host|custom` and how to interpret the reported `% error`.

## Scoring only

If you already have metrics CSVs:

- `pixi run paper-fidelity score --sim /abs/path/to/sim/request_metrics.csv --real /abs/path/to/real/request_metrics.csv`

Expected output:

- A report directory under `<WORKSPACE_ROOT>/results/reports/<date>/paper_fidelity/<scenario_name>/`

## Validation

Automated validation:

- Unit tests with fixed fixtures: `<WORKSPACE_ROOT>/tests/test_paper_fidelity_scorer.py`
- Run: `pixi run pytest tests`

Manual validation:

- Trace generation smoke: `pixi run python tests/manual/test_paper_fidelity_trace_smoke.py`
- Vidur simulation smoke: `pixi run python tests/manual/test_paper_fidelity_vidur_sim_smoke.py`
- Real replay smoke (GPU required): `pixi run python tests/manual/test_paper_fidelity_real_smoke.py`
- End-to-end smoke (GPU required): `pixi run python tests/manual/test_paper_fidelity_repro_smoke.py`
