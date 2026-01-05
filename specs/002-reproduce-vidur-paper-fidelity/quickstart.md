# Quickstart: Reproduce Vidur paper fidelity

**Spec**: `<WORKSPACE_ROOT>/specs/002-reproduce-vidur-paper-fidelity/spec.md`  
**Plan**: `<WORKSPACE_ROOT>/specs/002-reproduce-vidur-paper-fidelity/plan.md`  
**Date**: 2026-01-05

This quickstart describes the intended end-to-end workflow once the feature is implemented.

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

- `<WORKSPACE_ROOT>/configs/paper_fidelity/scenarios/llama2_7b_arxiv.yaml`

## End-to-end reproduction

Run **static** fidelity (paper “offline” workload):

- `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload static`

Run **dynamic** fidelity (paper “online” workload at 85% capacity):

- `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic`

Expected artifacts:

- Trace: `<WORKSPACE_ROOT>/tmp/paper_fidelity/traces/llama2_7b_arxiv/trace.csv`
- Sim metrics: `<WORKSPACE_ROOT>/tmp/paper_fidelity/runs/llama2_7b_arxiv/sim/request_metrics.csv`
- Real metrics: `<WORKSPACE_ROOT>/tmp/paper_fidelity/runs/llama2_7b_arxiv/real/request_metrics.csv`
- Report: `<WORKSPACE_ROOT>/results/reports/<date>/paper_fidelity/llama2_7b_arxiv/summary.md`

## Scoring only

If you already have metrics CSVs:

- `pixi run paper-fidelity score --sim /abs/path/to/sim/request_metrics.csv --real /abs/path/to/real/request_metrics.csv`

Expected output:

- A report directory under `<WORKSPACE_ROOT>/results/reports/<date>/paper_fidelity/<scenario_name>/`

## Validation

Planned automated validation:

- Unit tests with fixed fixtures: `<WORKSPACE_ROOT>/tests/test_paper_fidelity_scorer.py`

Planned manual validation:

- A single baseline scenario smoke run that produces a complete `summary.md` and required CSVs under `<WORKSPACE_ROOT>/tmp/paper_fidelity/`.
