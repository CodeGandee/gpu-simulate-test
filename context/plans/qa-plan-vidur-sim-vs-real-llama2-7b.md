# Q&A: plan-vidur-sim-vs-real-llama2-7b

## Introduction

This Q&A captures implementation questions for the paper-fidelity “Vidur sim vs Sarathi real” workflow for LLaMA2-7B, intended for developers (including future maintainers) running or extending the comparable-by-construction pipeline.

**Related docs**
- `context/plans/plan-vidur-sim-vs-real-llama2-7b.md`
- `context/tasks/working/req-vidur-sim-vs-real.md`
- `context/instructions/prep-dev-env.md`
- `specs/002-reproduce-vidur-paper-fidelity/quickstart.md`
- `specs/002-reproduce-vidur-paper-fidelity/qa-002-sim-vs-real-llama2-7b.md`

**Key entrypoints and modules**
- `configs/paper_fidelity/repro.yaml`
- `configs/paper_fidelity/trace.yaml`
- `configs/paper_fidelity/scale/small.yaml`
- `configs/paper_fidelity/scenario/llama2_7b_arxiv.yaml`
- `src/gpu_simulate_test/cli/paper_fidelity.py`
- `src/gpu_simulate_test/vidur_ext/sim_runner.py`
- `src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py`
- `src/gpu_simulate_test/paper_fidelity/scoring.py`
- `src/gpu_simulate_test/paper_fidelity/report.py`
- `src/gpu_simulate_test/paper_fidelity/manifest.py`
- `scripts/run_pf_llama2_7b_sim_vs_real.sh`

## How do I run Vidur simulation and Sarathi real inference for an apple-to-apple comparison of normalized latency?
> Last revised at: `2026-01-08T03:10:43Z` | Last revised base commit: `347ebf4ed2163cc82f0500b992ec30387b7c2264`

- Use the paper-fidelity entrypoint (it generates one canonical `trace.csv`, runs Vidur sim + Sarathi real on that same trace, then scores normalized metrics): `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload static` and `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic`.
- For faster iteration, add a scale preset: `--scale small|medium|full` (e.g. `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic --scale small`).
- Ensure prerequisites are met (Pixi + submodules + CUDA + model assets): `git submodule update --init --recursive`, `pixi install`, `pixi run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`, and `bash models/llama2-7b-hf/bootstrap.sh` (see `context/instructions/prep-dev-env.md`).
- Choose a Vidur profiling root:
  - Host bundle (recommended for meaningful gap on this host): set `scenario.vidur.profiling_root=results/raw/vidur-profiling/llama2-7b/sarathi-serve/latest` (or a timestamped directory under that tree).
  - Paper bundle (sanity-check pipeline): keep the scenario default `scenario.vidur.profiling_root=extern/tracked/vidur`.
  - Example override: `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload static --scale small scenario.vidur.profiling_root=results/raw/vidur-profiling/llama2-7b/sarathi-serve/latest`.
- Outputs to use for the normalized-latency comparison:
  - Shared trace: `tmp/paper_fidelity/traces/llama2_7b_arxiv/trace.csv`
  - Vidur sim request metrics: `tmp/paper_fidelity/runs/llama2_7b_arxiv/sim/request_metrics.csv` (normalized columns preserved from Vidur; see `src/gpu_simulate_test/vidur_ext/sim_runner.py`)
  - Sarathi real request metrics: `tmp/paper_fidelity/runs/llama2_7b_arxiv/real/request_metrics.csv` (Sarathi in-engine metrics; see `src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py`)
  - Report (includes ECDF SVGs for normalized latency): `results/reports/<date>/paper_fidelity/<scenario_name>/summary.md` and `scores.json` (see `src/gpu_simulate_test/paper_fidelity/report.py`).
- Apple-to-apple guards are enforced (hard failures): Sarathi must decode exactly `num_decode_tokens` per request (`scenario.real.sampling.ignore_eos=true`), Vidur `request_num_decode_tokens` must match the same trace, and sim-vs-real request ids + token counts must match before scoring (`src/gpu_simulate_test/paper_fidelity/scoring.py`).

## We have small/medium/full dataset splits; how do I use them?
> Last revised at: `2026-01-08T03:14:38Z` | Last revised base commit: `347ebf4ed2163cc82f0500b992ec30387b7c2264`

- Use the built-in `scale` config group via CLI: `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload static --scale small` (or `medium`, `full`).
- The scale presets map to `trace_subset` defaults:
  - `small`: first 50 requests (`trace_subset.kind=range trace_subset.begin=0 trace_subset.end=50`) from `configs/paper_fidelity/scale/small.yaml`
  - `medium`: first 500 requests (`trace_subset.kind=range ... end=500`) from `configs/paper_fidelity/scale/medium.yaml`
  - `full`: all requests (`trace_subset.kind=all`) from `configs/paper_fidelity/scale/full.yaml`
- You can also use it as a Hydra override (equivalent to `--scale`): `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic scale=small`.
- If you need a custom slice, bypass the preset by overriding `trace_subset.*` directly (this wins over the scale defaults): `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic scale=small trace_subset.end=32`.
- The batch script uses these presets automatically: `bash scripts/run_pf_llama2_7b_sim_vs_real.sh` (runs static+dynamic across `small|medium|full` and writes a `manifest.json`).

## In Vidur simulation and Sarathi-Serve inference, can we separately benchmark prefill vs decode for sim-vs-real stats?
> Last revised at: `2026-01-08T05:18:23Z` | Last revised base commit: `ce3d0a72e4174572246e25c9105039f353d7a83e`

- Yes: both sides already emit prefill- and decode-scoped request metrics (including normalized variants) in the same per-run CSVs used by paper-fidelity.
- Use these columns (examples): `prefill_time_execution_plus_preemption_normalized` and `decode_time_execution_plus_preemption_normalized` (and, where present, `prefill_e2e_time_normalized`).
- Where to read them:
  - Vidur sim: `tmp/paper_fidelity/runs/<scenario.name>/sim/request_metrics.csv` (source is `tmp/paper_fidelity/runs/<scenario.name>/sim/vidur_raw/<timestamp>/request_metrics.csv` via `src/gpu_simulate_test/vidur_ext/sim_runner.py`).
  - Sarathi real: `tmp/paper_fidelity/runs/<scenario.name>/real/request_metrics.csv` (source is `tmp/paper_fidelity/runs/<scenario.name>/real/sarathi/replica_0/sequence_metrics.csv` via `src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py`).
- The paper-fidelity scoring/report now includes these stage metrics by default (alongside request-level metrics); see the metric list in `src/gpu_simulate_test/cli/paper_fidelity.py` (`_run_score_only`) and the generated `results/reports/<date>/paper_fidelity/<scenario>/scores.json`.
- Keep decode-length semantics fixed for comparability: `scenario.real.sampling.ignore_eos=true` (default) and token-count validators must pass (see `src/gpu_simulate_test/paper_fidelity/scoring.py`).

## Does `request_execution_plus_preemption_time_normalized` include both prefill and decode?
> Last revised at: `2026-01-08T04:30:39Z` | Last revised base commit: `347ebf4ed2163cc82f0500b992ec30387b7c2264`

- Yes: it is a whole-request metric that spans from first schedule time to completion, so it includes **prefill + decode**, plus any “preempted/bubble” time after the request is scheduled.
- In Vidur, it is computed as `(request.execution_time + request.preempted_time) / request.num_decode_tokens` (`extern/tracked/vidur/vidur/metrics/metrics_store.py`), which matches the definition “exclude initial scheduling delay” (`c_r - s_r`) and then normalize by output (decode) tokens (`extern/tracked/vidur/docs/metrics.md`).
- In Sarathi-Serve, it follows the same metric vocabulary/definitions (`c_r - s_r` normalized by output tokens) (`extern/tracked/sarathi-serve/sarathi/metrics/README.md`).
- If you want stage-separated metrics instead, use:
  - `prefill_time_execution_plus_preemption_normalized = (f_r - s_r) / num_prefill_tokens`
  - `decode_time_execution_plus_preemption_normalized = (c_r - f_r) / num_decode_tokens`
