# Requirements: Vidur simulation vs real inference comparison (paper-fidelity comparable experiments)

## HEADER
- **Purpose**: Define requirements for running *comparable* experiments when comparing Vidur (CPU-side simulator) against a real inference engine, so sim vs real latency distributions and derived metrics are meaningful.
- **Status**: Working draft
- **Date**: 2026-01-07
- **Primary reference (authoritative)**: `specs/002-reproduce-vidur-paper-fidelity/` and `configs/paper_fidelity/` (current paper-fidelity workflow and design choices).
- **Secondary reference (may be outdated; treat as API clues only)**: `specs/001-compare-vidur-real-timing/` and `configs/compare_vidur_real/` (earlier sim-vs-real timing harness; useful for locating APIs and prior patterns).
- **Related implementations**:
  - Paper-fidelity pipeline (paper-aligned normalized metrics): `src/gpu_simulate_test/cli/paper_fidelity.py`, `src/gpu_simulate_test/paper_fidelity/`, `src/gpu_simulate_test/vidur_ext/sim_runner.py`, `src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py`
  - API clues (legacy compare tooling; not a design reference): `src/gpu_simulate_test/real_bench/`, `src/gpu_simulate_test/analysis/`
- **Related runbooks**:
  - Env: `context/instructions/prep-dev-env.md`
  - Paper fidelity quickstart: `specs/002-reproduce-vidur-paper-fidelity/quickstart.md`
- **Assumed inputs for this run (simplified TP=1/PP=1)**:
  - Model: `meta-llama/Llama-2-7b-hf` (assets at `models/llama2-7b-hf/source-data`)
  - Vidur profiling root (microbench results): `results/raw/vidur-profiling/llama2-7b/sarathi-serve/latest`
  - Trace lengths source (untimed): `extern/tracked/vidur/data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv`
  - Real inference backend: Sarathi-Serve (`extern/tracked/sarathi-serve/`) via `src/gpu_simulate_test/real_bench/backends/sarathi_paper_fidelity_backend.py`

---

## 1) Problem statement (what “comparable” means)

Vidur simulation outputs are only comparable to real inference results if **the two runs represent the same workload and the same system boundary**. “Comparable” requires:

1. **Same request shapes**: prefill token count and decode token count per request must match the simulator’s assumptions.
2. **Same arrival process**: requests must enter the system with the same schedule and concurrency expectations.
3. **Same topology and parallelism**: TP/PP, device class, and network topology assumptions must match (or be explicitly noted).
4. **Same scheduling boundary**: timestamps/metrics must measure equivalent phases (queueing, prefill, decode, preemption).
5. **Same stopping behavior**: EOS/stop conditions must not silently truncate decode lengths unless both sides account for it consistently.

If any of these invariants differ, “sim vs real gap” is not interpretable as simulator fidelity; it becomes a mixture of modeling mismatch, runtime mismatch, and boundary mismatch.

---

## 2) Comparison mode in scope: paper-fidelity (paper-aligned normalized metrics)

Goal: compare paper-defined normalized request metrics (static + dynamic at 85% capacity), and optionally compare against paper reference values.

- Workload input: canonical `trace.csv` (`arrived_at,num_prefill_tokens,num_decode_tokens`) used by:
  - Vidur trace-driven simulation (`TraceRequestGeneratorConfig`)
  - Sarathi-Serve real replay with synthetic `prompt_token_ids` (avoids tokenization variability)

Comparability sensitivity:
- Real replay uses Sarathi’s in-engine metrics to align timing boundaries with Vidur definitions.
- Dynamic mode includes capacity discovery; the 85% operating point must be computed consistently.

Primary references:
- `specs/002-reproduce-vidur-paper-fidelity/spec.md`
- `src/gpu_simulate_test/cli/paper_fidelity.py`

---

## 3) Hard requirements (must-haves)

### R1. Single canonical workload representation

The experiment MUST have one source of truth for request shapes and arrivals, then derive both sim and real inputs from it:

- `tmp/paper_fidelity/traces/<scenario>/trace.csv`

Avoid running “two independent Poisson draws” for sim vs real; that produces different queueing dynamics.

### R2. Deterministic seeds and provenance

The run MUST record the exact resolved config + seeds + code version:

- Run inside Pixi (`pixi run ...`)
- Record `git commit` + `git dirty` + environment snapshot
- Persist metadata (`run_meta.json`, `trace_meta.json`, etc.)

### R3. Matching decode length semantics

The run MUST make decode length behavior explicit:

- Vidur simulation and real inference (Sarathi-Serve) MUST produce the **same number of decode tokens per request**.
- Real replay MUST be configured to prevent early termination (e.g., `ignore_eos: true`) and MUST validate that the decoded token counts match the trace.
- If the decoded token counts do not match, treat the run as invalid (do not “paper over” the mismatch by truncation for this workflow).

Paper-fidelity requirement: use `ignore_eos: true` for Sarathi and validate decoded token counts.

### R4. Timing boundary alignment

The compared metrics MUST represent the same boundary:

- Compare Vidur/Sarathi *normalized metrics* (e.g., `request_e2e_time_normalized`) without recomputation.

Do not compare a client-side “wall clock” latency against a simulator metric that excludes queueing/preemption, unless you explicitly transform one side to match the other.

### R5. Simulator profiling provenance is explicit

Vidur simulation MUST run with a clearly defined profiling root:

- For this run, use the host-collected microbenchmark profiling root:
  - `results/raw/vidur-profiling/llama2-7b/sarathi-serve/latest`
- Any other profiling root MUST be called out in reports/metadata because it changes how to interpret `% error`.

Practical notes (from `specs/002-reproduce-vidur-paper-fidelity/qa-002-sim-vs-real-llama2-7b.md`):

- `latest/` is a *local symlink* to the newest timestamped profiling run directory. For reproducibility, prefer pinning the timestamped directory (and/or record the exact `profiling_meta.json` contents).
- This host profiling bundle is **compute-only** by default (no network profiling, no CPU overhead profiling). That is fine for TP=1/PP=1 comparisons as long as the simulation config matches those modeling choices (e.g., skip CPU overhead modeling if the profiling root does not contain CPU overhead CSVs).
- Profiling bundles are parameterized (grid size, TP degree, max tokens, etc.). This bundle is TP=1; simulations at other TP degrees require re-profiling with matching knobs.

### R6. Report artifacts (JSON + markdown + SVG)

Each reproduction SHOULD produce:

- A machine-readable JSON summary for programmatic processing:
  - `results/reports/<date>/paper_fidelity/<scenario>/scores.json`
  - Includes per-metric percentile summaries and percent error for sim vs real (and sim vs paper when paper reference is enabled).
- A human-readable markdown summary:
  - `results/reports/<date>/paper_fidelity/<scenario>/summary.md`
  - Must include a percentile table for **static** and **dynamic** normalized latency metrics.
- SVG figures embedded in the markdown:
  - ECDF plots (sim vs real) for:
    - Static: `request_execution_plus_preemption_time_normalized`
    - Dynamic: `request_e2e_time_normalized`
  - Percentile bar plots (sim vs real, and optionally paper) using the configured percentiles (default: P50/P95) for the same two metrics.

---

## 4) Setup requirements (environment + dependencies)

### S1. Environment

- Initialize submodules: `git submodule update --init --recursive`
- Create Pixi env: `pixi install`
- GPU sanity check (required for real runs): `pixi run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
- Multi-GPU note (this host): when `CUDA_VISIBLE_DEVICES` is not set, the Sarathi paper-fidelity runner selects the **MIG-disabled GPU with the least used VRAM** by default; override with `CUDA_VISIBLE_DEVICES=<idx>` if needed.

See `context/instructions/prep-dev-env.md`.

### S2. Model assets (external references)

- Ensure model symlinks exist under `models/*/source-data` using `bash models/bootstrap.sh` or per-model bootstrap scripts.

For this run, use:

- Model assets: `models/llama2-7b-hf/source-data` (`meta-llama/Llama-2-7b-hf`)
- Trace lengths source: `extern/tracked/vidur/data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv`
- Vidur profiling root: `results/raw/vidur-profiling/llama2-7b/sarathi-serve/latest`

Dataset note (from Q&A): you do **not** need the original text dataset for this trace; the processed trace CSV already contains the token-length distribution (`num_prefill_tokens`, `num_decode_tokens`), and this workflow replays *lengths*, not raw prompts.

---

## 5) Experiment design requirements (make the two sides match)

### D1. Fix the “slice” (hardware + scheduler knobs)

This run uses a simplified single-replica, no-parallelism setup:

- `tensor_parallel_size = 1`
- `pipeline_parallel_size = 1` / `num_pipeline_stages = 1`

Record and keep constant (within this simplified setting):

- GPU type and memory class (e.g., A100 80GB)
- Network topology assumption for Vidur (`network_device`)
- Real engine scheduler knobs (batch caps, chunk size, max sequences, etc.)

If hardware or topology mismatches exist, treat results as *qualitative* and document the mismatch.

#### D1.1 Explicitly set parity-critical knobs (do not trust defaults)

For sim-vs-real fidelity work, **do not rely on default values** in Vidur, Sarathi-Serve, or the profiling bundle selection. Set them explicitly in config/CLI overrides and record them in metadata.

Rationale:
- Defaults differ across systems (and can change across versions) while producing “plausible” metrics.
- We have observed cases where the main request metric looks moderately close while the **prefill vs decode split is badly wrong** due to knob mismatch, e.g. `chunk_size` and batch caps (see `context/issues/known/issue-vidur-prefill-underestimate-decode-overestimate.md`).

At minimum, explicitly align and record the following:

- **Scheduler type**
  - Real: Sarathi-Serve scheduler used by the backend (paper-fidelity runner).
  - Sim: Vidur `replica_scheduler_config` type (e.g. `sarathi`, `vllm`, etc.).
  - Note: “same metric name” does not imply “same scheduler behavior”; stage-level metrics are especially sensitive.
- **Chunking / iteration limits**
  - Real: `scenario.real.scheduler.chunk_size`
  - Sim (Sarathi scheduler): Vidur `sarathi_scheduler_config.chunk_size`
  - Do not assume Vidur defaults match real. Vidur’s Sarathi `chunk_size` default is much larger (see `extern/tracked/vidur/vidur/config/config.py`).
- **Batch caps / concurrency**
  - Real: `scenario.real.scheduler.max_num_seqs` (and any other max-inflight controls)
  - Sim: Vidur `replica_scheduler_config.batch_size_cap` (and any per-iteration token caps for vLLM/LightLLM schedulers)
  - Ensure the mapping is intentional and documented (e.g. for TP1/PP1, `batch_size_cap ≈ max_num_seqs` is a reasonable starting point).
- **Parallelism and topology**
  - Real: `scenario.real.parallel.tensor_parallel_size`, `scenario.real.parallel.pipeline_parallel_size`
  - Sim: `scenario.vidur.tensor_parallel_size`, `scenario.vidur.num_pipeline_stages`, `scenario.vidur.network_device`, `scenario.vidur.device`
  - Profiling root must match these (profiling files are indexed by device/model/topology and are not universally interchangeable).
- **Max tokens / trace bounds**
  - Real + sim must agree on `max_tokens` and trace generation constraints, otherwise batching/eviction behavior can diverge.
- **CPU overhead modeling**
  - Sim: `scenario.vidur.skip_cpu_overhead_modeling` must be set explicitly.
  - Profiling: if CPU overhead modeling is enabled, the profiling root must include CPU-overhead measurements consistent with the real stack; otherwise comparisons are not meaningful.

Implementation note (this repo’s paper-fidelity pipeline):
- The paper-fidelity Vidur runner reads `scenario.vidur.scheduler.*` and constructs Vidur’s `replica_scheduler_config` explicitly; keep these keys set in scenario configs and do not rely on Vidur’s internal defaults.

### D2. Make arrivals identical

Pick one:

- Deterministic schedule (recommended for repeatability):
  - Use a fixed `arrived_at` schedule in the derived `trace.csv`.
- Seeded Poisson arrivals (recommended for paper-fidelity dynamic workloads):
  - Generate arrivals once with a fixed seed, and reuse the same derived trace for both sim and real.

Dynamic pipeline note (from Q&A):

- Poisson arrivals are generated via exponential inter-arrivals with mean `1/qps`, using `workload.seed` for determinism (`src/gpu_simulate_test/paper_fidelity/traces.py:add_poisson_arrivals()`).
- For the baseline scenario, dynamic runs first perform **capacity discovery** on Sarathi and then set the workload QPS to `qps_85 = 0.85 * capacity_qps` (default overload criterion: P99(`request_scheduling_delay`) > 5s). The final dynamic `trace.csv` is regenerated using that `qps_85` and `workload.seed`.
- Capacity search writes an intermediate `tmp/paper_fidelity/runs/<scenario>/capacity/trace.csv` (overwritten across candidates); the final shared trace is `tmp/paper_fidelity/traces/<scenario>/trace.csv`.

### D3. Warm up and reset metrics

Both systems SHOULD run a warmup and clear metrics before measuring:

- Sarathi paper-fidelity runner warms up and calls `engine.reset_metrics()` before replay.
- If you change the real runner, replicate the same pattern; otherwise, the first batch includes compilation/caching effects.

Warm-up requirement (from Q&A): the Sarathi paper-fidelity path includes an explicit warm-up request and metric reset. This is required to avoid “first request” artifacts contaminating tail latency distributions.

### D4. Avoid tokenization variability (prefer token-length replay when possible)

For best comparability, drive real inference by **token IDs** rather than raw text:

- Paper-fidelity Sarathi replay uses synthetic `prompt_token_ids` of length `num_prefill_tokens`.
- This isolates scheduling/runtime behavior from tokenization and prompt content effects.

### D5. Trace subsetting for faster iteration (must remain consistent)

If you subset a trace to speed up iteration, apply it **once** at the shared trace layer so sim and real see the same requests:

- Paper fidelity: `trace_subset.kind=range` works for timed and untimed sources; `indices` is only allowed for untimed sources.
- Always keep ordering stable; do not reorder requests.

---

## 6) Acceptance criteria (what to check after a run)

### A) Basic sanity

- Both sides consumed the same number of requests (after subsetting, if used).
- For fixed-length decode runs, `actual_num_decode_tokens == expected_num_decode_tokens` for all requests (paper-fidelity enforces this).
- Arrival schedule monotonic and non-negative (trace validators enforce this).

### B) Metric comparability

- The compared metrics correspond to the same definition:
  - normalized execution/e2e metrics vs the same normalized metrics (from Vidur/Sarathi)
- Any unavoidable boundary mismatch is explicitly documented in metadata/report.
- Key knobs that affect metric boundaries (scheduler type, chunk size, batch caps, TP/PP) are recorded and confirmed aligned; if not aligned, the run is labeled “not comparable”.

### C) Provenance completeness

- Output directories contain `run_meta.json` (and `trace_meta.json` for trace generation) with:
  - resolved configs, seeds, git commit, env snapshot, and key artifact paths.

---

## 7) Practical “do this” templates

### Paper fidelity (recommended)

Use the baseline scenario (`configs/paper_fidelity/scenario/llama2_7b_arxiv.yaml`) and override the profiling root to the host microbenchmark bundle:

- `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload static scenario.vidur.profiling_root=$(pwd)/results/raw/vidur-profiling/llama2-7b/sarathi-serve/latest`
- `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic scenario.vidur.profiling_root=$(pwd)/results/raw/vidur-profiling/llama2-7b/sarathi-serve/latest`

Fast iteration:

- `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic scenario.vidur.profiling_root=$(pwd)/results/raw/vidur-profiling/llama2-7b/sarathi-serve/latest trace_subset.kind=range trace_subset.begin=0 trace_subset.end=32`

---

## Next steps

- Add a small checklist to runbooks for “comparability invariants” (arrivals, decode lengths, profiling mode, scheduler knobs).

---

## Appendix: Standard experiment scales (trace subset sizes)

To make iteration predictable, standardize three “scales” for the same scenario by subsetting the
input trace rows (row index order preserved):

- **Small**: first 50 requests (`trace_subset.kind=range trace_subset.begin=0 trace_subset.end=50`)
- **Medium**: first 500 requests (`trace_subset.kind=range trace_subset.begin=0 trace_subset.end=500`)
- **Full**: all requests (`trace_subset.kind=all`)

Example (dynamic workload; always keep sim + real on the same subset):

- Small:
  - `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic scenario.vidur.profiling_root=$(pwd)/results/raw/vidur-profiling/llama2-7b/sarathi-serve/latest trace_subset.kind=range trace_subset.begin=0 trace_subset.end=50`
- Medium:
  - `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic scenario.vidur.profiling_root=$(pwd)/results/raw/vidur-profiling/llama2-7b/sarathi-serve/latest trace_subset.kind=range trace_subset.begin=0 trace_subset.end=500`
- Full:
  - `pixi run paper-fidelity repro --scenario llama2_7b_arxiv --workload dynamic scenario.vidur.profiling_root=$(pwd)/results/raw/vidur-profiling/llama2-7b/sarathi-serve/latest trace_subset.kind=all`
