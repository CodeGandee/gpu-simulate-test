# Vidur stage breakdown skew: prefill too fast, decode too slow (static workloads)

## Summary

In our **static** paper-fidelity sim-vs-real runs (Vidur sim vs Sarathi-Serve real), Vidur can produce a **misleadingly “okay” total service-time metric** while the **prefill vs decode breakdown is badly skewed**:

- `prefill_time_execution_plus_preemption_normalized` is **vastly underpredicted** (Vidur ≪ real).
- `decode_time_execution_plus_preemption_normalized` is **materially overpredicted** (Vidur ≫ real).
- The main paper-facing static metric `request_execution_plus_preemption_time_normalized` can look much closer because these two errors partially cancel.

This is a red flag that the simulator is attributing too little time to “time-to-first-token work” and too much time to “per-token decode work” under our static configuration.

## Background (what these metrics mean)

Vidur defines these request-level stage metrics (see `extern/tracked/vidur/docs/metrics.md` and `extern/tracked/vidur/vidur/metrics/metrics_store.py`):

- `prefill_time_execution_plus_preemption_normalized`:
  - `(prefill_completed_at - scheduled_at) / num_prefill_tokens`
  - “Time-to-first-token after admission”, normalized by prompt tokens.
- `decode_time_execution_plus_preemption_normalized`:
  - `(completed_at - prefill_completed_at) / num_decode_tokens`
  - “Post-first-token time”, normalized by output tokens.

Both include “preemption time” in Vidur’s sense: time a request is allocated but not executing (pipeline bubbles, time between iterations, etc.).

## Observation (evidence)

Two concrete examples (LLaMA2-7B, Arxiv-4K):

### Static small

Report:
- `results/reports/2026-01-08/paper_fidelity/llama2_7b_arxiv_sim_vs_real_2026-01-08_03-43-35-976644797_static_small/summary.md`

P50/P95 (Vidur sim vs Sarathi real):
- Prefill normalized:
  - P50: `0.0000999` vs `0.001114` (≈ **11× lower**)
  - P95: `0.0001297` vs `0.001270` (≈ **9.8× lower**)
- Decode normalized:
  - P50: `0.03827` vs `0.01690` (≈ **2.3× higher**)
  - P95: `0.04412` vs `0.01802` (≈ **2.4× higher**)

Yet the main static metric is much closer:
- `request_execution_plus_preemption_time_normalized` P50: `0.04006` vs `0.03554` (≈ **+12.7%**)

### Static medium

Report:
- `results/reports/2026-01-08/paper_fidelity/llama2_7b_arxiv_sim_vs_real_2026-01-08_04-26-21-620353247_static_medium/summary.md`

P50/P95 (Vidur sim vs Sarathi real):
- Prefill normalized:
  - P50: `0.0001338` vs `0.001191` (≈ **8.9× lower**)
  - P95: `0.0001865` vs `0.001456` (≈ **7.8× lower**)
- Decode normalized:
  - P50: `0.05322` vs `0.01708` (≈ **3.1× higher**)
  - P95: `0.05504` vs `0.01879` (≈ **2.9× higher**)

## Likely causes (hypotheses)

### 1) Sim/real scheduler knobs don’t match (chunk size + batch cap)

This is the most concrete/likely cause for the **direction** of the skew.

- Real (Sarathi) run uses:
  - `chunk_size=16`
  - `max_num_seqs=16`
  - See the run metadata in the report dir, e.g. `.../run_meta.json` (`params.scenario.real.scheduler`).
- Vidur sim (via our wrapper `src/gpu_simulate_test/vidur_ext/sim_runner.py`) does **not** set Vidur’s `replica_scheduler_config`, so Vidur uses its defaults:
  - `ClusterConfig.replica_scheduler_config = SarathiSchedulerConfig(chunk_size=512, batch_size_cap=128, ...)`
  - See `extern/tracked/vidur/vidur/config/config.py` (`ClusterConfig` + `SarathiSchedulerConfig` defaults).

Why that matches the observed sign under **static arrivals**:
- A **larger chunk size** reduces the number of “prefill scheduling boundaries” per request → tends to make prefill complete “too quickly” in the sim (lower prefill normalized time).
- A **larger chunk size and/or larger batch cap** increases per-iteration work and can increase gaps between a request’s decode iterations → tends to inflate decode normalized time.

Vidur’s own README example also uses Sarathi chunking knobs explicitly (`--sarathi_scheduler_config_chunk_size 512`), and the paper discusses Sarathi chunk sizes in the 512/1K/2K range (see `extern/tracked/vidur/paper/tex/5-eval.tex`), so `chunk_size=16` may be far outside the regime Vidur’s defaults were intended to mirror.

### 2) Vidur’s Sarathi scheduler model is intentionally simplified

The paper notes the batching policies are implemented in “less than 150 lines of Python code” (see `extern/tracked/vidur/paper/tex/3-design.tex`), which is great for extensibility but can miss real-system details that affect prefill/decode overlap (especially under static arrivals).

If Vidur’s Sarathi model differs from Sarathi-Serve’s exact batching behavior (microbatching, fairness between prefills/decodes, etc.), the prefill/decode split can drift even if totals look similar.

### 3) CPU/runtime overhead is excluded in the sim (prefill-heavy bias)

Our runs commonly keep CPU overhead modeling disabled on the Vidur side (`scenario.vidur.skip_cpu_overhead_modeling=true`).

Vidur’s docs explicitly call out CPU overhead profiling as a separate component (`extern/tracked/vidur/docs/profiling.md`). If real prefill has non-trivial host overhead (scheduler bookkeeping, sampling setup, synchronization, etc.), the sim can underpredict prefill more than decode.

This doesn’t by itself explain the decode *over*prediction, but it can amplify the “prefill too fast” half of the skew.

### 4) Execution-time predictor / profiling coverage mismatch

Vidur uses a random-forest predictor over profiled attention/MLP kernels (paper §3; `extern/tracked/vidur/paper/tex/3-design.tex`).

If the profiling bundle and predictor aren’t well-covered for our effective shapes (especially with `chunk_size=16`, small batch caps, or different kernel implementations), the sim can misattribute work between stages.

## How to address

### A) Make sim and real scheduler knobs comparable (recommended first step)

Update the Vidur invocation to explicitly set scheduler knobs to match the real run:

- `chunk_size` (Sarathi) ↔ `scenario.real.scheduler.chunk_size`
- `batch_size_cap` (Vidur) ↔ `scenario.real.scheduler.max_num_seqs` (or an intentionally chosen mapping)

Concretely, in our integration this likely means:
- Extend `src/gpu_simulate_test/vidur_ext/sim_runner.py` (`run_vidur_paper_fidelity_sim`) to construct `ClusterConfig(..., replica_scheduler_config=SarathiSchedulerConfig(...))` from the scenario config, instead of relying on Vidur defaults.
- Record the chosen Vidur scheduler config into `tmp/paper_fidelity/.../sim/run_meta.json` so reports can sanity-check sim/real parity.

Status:
- The paper-fidelity pipeline now supports explicit Vidur scheduler knobs via `scenario.vidur.scheduler.*` and passes them into Vidur’s `replica_scheduler_config`.

Expected outcome:
- Prefill normalized time should increase toward real (more chunk boundaries).
- Decode normalized time should decrease toward real (less “decode blocked by giant prefill chunks / oversized batches”).

### B) Add a “config parity” guardrail in reporting

In `paper_fidelity` report metadata, store and/or validate:
- Vidur scheduler type + `chunk_size` + `batch_size_cap`
- Real scheduler type + `chunk_size` + `max_num_seqs`

If they don’t match, annotate the report as “not stage-comparable”.

### C) If skew remains after knob parity: investigate model/overhead fidelity

Next diagnostic steps:
- Re-run with `scenario.vidur.skip_cpu_overhead_modeling=false` using a host-profiled `cpu_overheads.csv` (Vidur supports this; see `extern/tracked/vidur/docs/profiling.md`).
- Enable richer Vidur metric outputs (token completion / batch metrics) to confirm whether the skew is driven by long decode-iteration gaps vs kernel-time prediction error.
- Validate Sarathi-Serve’s metric definitions for `prefill_completed_at` vs Vidur’s (`prefill_completed_at` is a common boundary but can differ subtly across implementations).
