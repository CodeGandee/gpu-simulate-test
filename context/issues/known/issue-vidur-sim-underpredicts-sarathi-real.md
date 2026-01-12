# Issue: Vidur sim underpredicts Sarathi real (paper-fidelity LLaMA2-7B; CPU overhead profiling/modeling)

**Status**: Known (guardrails implemented; needs re-run with real CPU overhead inputs)
**Date**: 2026-01-09
**Last updated**: 2026-01-12
**Priority**: High (Fidelity gap)

## Summary

In paper-fidelity experiments for **LLaMA2-7B (TP=1) on A100**, Vidur simulation underpredicts Sarathi-Serve real latency.

This issue combines two observations that looked like separate problems but share the same root cause:

1) A large underprediction (~25%) when CPU overhead modeling is effectively **disabled / missing inputs**.
2) A smaller but still material underprediction (~16–20%) when CPU overhead modeling was “enabled” but the CPU overhead inputs were actually **dummy/unprofiled** (because profiling failed).

## Observation A: CPU overhead modeling disabled / missing inputs (~25%)

Example report:

- `results/reports/2026-01-08/paper_fidelity/llama2_7b_arxiv_sim_vs_real_2026-01-08_12-22-51-887166845_static_medium`

| Metric | Sim (P50) | Real (P50) | % Error |
| :--- | :--- | :--- | :--- |
| `request_execution_plus_preemption_time_normalized` | 0.027 | 0.036 | **24.58%** |
| `request_e2e_time_normalized` | 3.00 | 3.94 | **23.89%** |

## Observation B: “CPU overhead modeling enabled” but still ~16–20%

Report:

- `results/reports/2026-01-09/paper_fidelity/llama2_7b_arxiv/summary.md`

The run was configured with `skip_cpu_overhead_modeling=false`, yet sim still underpredicts by ~16–20% across both request-level and stage-level metrics.

Key tell: prefill and decode have similar percent error, suggesting a global underprediction rather than a scheduler-skew mismatch.

## Key finding: the CPU overhead inputs were dummy/unprofiled

The “CPU overhead enabled” report used a custom profiling root:

- Profiling root: `tmp/test_profiling_root`

The staged CPU overhead CSV exists but is clearly synthetic:

- `tmp/test_profiling_root/data/profiling/cpu_overhead/a100_pairwise_nvlink/meta-llama/Llama-2-7b-hf/cpu_overheads.csv`
- Every overhead column is constant across rows (matches `tests/manual/generate_dummy_cpu_overhead.py`).
- `tmp/test_profiling_root/profiling_meta.json` reports `cpu_overhead_profiled=false` and `cpu_overheads_csv=null` under `profiling_outputs`.

So “enabled CPU overhead modeling” was effectively “enabled against placeholder inputs”.

## Root cause: CPU overhead profiling failed on this host

Host profiling produced an empty `cpu_overhead.csv` (0 rows / 1 byte) and never staged a real `cpu_overheads.csv` into a host profiling root under `tmp/paper_fidelity/profiling_roots/...`.

Evidence captured in the original investigation:

- CPU overhead profiler output exists but is effectively empty:
  - `tmp/paper_fidelity/profiling_outputs/llama2_7b_arxiv/2026-01-09_06-26-32-853654/cpu_overhead/2026-01-09_06-28-24/meta-llama/Llama-2-7b-hf/cpu_overhead.csv`
- Ray logs showed Torch CUDA init failing with an invalid device index when all GPUs were visible.

Hypothesis: one GPU is in a bad state / MIG-visible such that Torch’s enumeration fails inside Ray workers unless we explicitly pin a known-good subset.

## Mitigations implemented (guardrails + ergonomics)

These changes prevent “silent success” with empty/dummy CPU overhead inputs and surface CPU overhead status in reports:

- **Fail fast** if CPU overhead profiling would produce 0 rows:
  - `src/gpu_simulate_test/vidur_ext/vidur_profiling_cpu_overhead_main.py`
- **Validate CPU overhead CSVs** (non-empty, required columns, and placeholder-like detection):
  - `src/gpu_simulate_test/vidur_ext/cpu_overhead_validation.py`
  - Used both when staging outputs and when consuming a profiling root.
- **Profiling-root guardrail**: when `skip_cpu_overhead_modeling=false`, validate `cpu_overheads.csv` content (strict by default):
  - `src/gpu_simulate_test/vidur_ext/profiling_root.py`
- **Report annotation**: `summary.md` includes a `Profiling -> cpu_overhead` status block (enabled/disabled, csv path, validation status, warnings):
  - `src/gpu_simulate_test/paper_fidelity/report.py`
  - `src/gpu_simulate_test/cli/paper_fidelity.py`
- **Config knobs for fast iteration**:
  - `profiling.cpu_overhead.max_batch_size` and `profiling.cpu_overhead.validation` in `configs/paper_fidelity/profile.yaml`
  - `scenario.vidur.cpu_overhead.validation` in `configs/paper_fidelity/scenario/llama2_7b_arxiv.yaml`

## How to reproduce and fix (recommended path)

1) Pin a healthy GPU subset via repo-local `.env`:
   - `export GSIM_CUDA_VISIBLE_DEVICES=0`
   - See `context/instructions/prep-dev-env.md`
2) Re-run host profiling with CPU overhead enabled (bounded for smoke):
   - `pixi run paper-fidelity profile --scenario llama2_7b_arxiv --include-cpu-overhead profiling.cpu_overhead.max_batch_size=16`
3) Re-run repro with CPU overhead modeling enabled and using the host profiling root:
   - `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload static scenario.vidur.profiling_root=<profiling_root> scenario.vidur.skip_cpu_overhead_modeling=false`
4) Inspect the report’s `Profiling -> cpu_overhead` block to confirm inputs are “ok” (not missing/placeholder).

## If a residual gap remains after real CPU overhead profiling works

If, after using a real `cpu_overheads.csv`, a ~10–20% gap persists:

- Confirm compute/attention profiling did not fall back to templates and matches Sarathi’s backend/kernel path.
- Capture real-side provenance (backend, dtype, cache settings, CUDA graphs, etc.) and verify profiling/real parity.
- Consider that Vidur’s CPU overhead model covers specific sections and may miss additional real stack costs (Ray jitter, Python overhead beyond Vidur’s metrics).
