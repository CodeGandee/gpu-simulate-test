# Data Model: Reproduce Vidur paper fidelity

**Spec**: `<WORKSPACE_ROOT>/specs/002-reproduce-vidur-paper-fidelity/spec.md`  
**Plan**: `<WORKSPACE_ROOT>/specs/002-reproduce-vidur-paper-fidelity/plan.md`  
**Date**: 2026-01-05

This feature is CLI- and artifact-driven (no database). The “data model” is the set of configs and on-disk artifacts that must be produced and consumed reproducibly.

**Path convention**: `<WORKSPACE_ROOT>` refers to the repository root.

## Entities

### Scenario

Represents one paper-fidelity reproduction target (model + workload + hardware + backend knobs).

**Identity**

- `scenario_name: str` (stable identifier used in paths and reports; e.g., `llama2_7b_arxiv_a100_tp1`)

**Config fields (logical)**

- `model_id: str` (e.g., `meta-llama/Llama-2-7b-hf`)
- `model_ref: str | Path` (local model reference for real runs; e.g., `<WORKSPACE_ROOT>/models/llama2-7b-hf/source-data`)
- `hardware_id: str` (e.g., `a100`)
- `trace_source`:
  - `kind: enum {vidur_processed_lengths_csv, trace_csv, legacy_workload_dir}`
  - `path: Path`
  - `max_tokens: int` (default 4096; matches paper’s truncation)
  - `num_requests: int | null` (optional cap; if null, consume all rows)
  - `seed: int` (deterministic sampling/shuffle when applicable)
- `workload`:
  - `mode: enum {static, dynamic}`
  - `qps: float` (dynamic only; used for Poisson arrivals)
  - `seed: int` (dynamic only; Poisson generator seed)
- `vidur` (simulation):
  - `profiling_root: Path` (default uses `<WORKSPACE_ROOT>/extern/tracked/vidur/data/profiling` or a scenario override)
  - `model_key: str` (Vidur model key, if needed by wrappers)
  - scheduler knobs needed for paper alignment (TP/PP, batch caps, etc.)
- `real`:
  - `backend: enum {sarathi}` (MVP)
  - scheduler knobs (chunk size, max_num_seqs, token caps, etc.)
- `capacity_search`:
  - `enabled: bool`
  - `min_qps: float`, `max_qps: float` (search range)
  - `max_iters: int` (binary search limit)
  - `overload_p99_scheduling_delay_s: float` (default 5.0)
  - `qps_operating_point_fraction: float` (default 0.85)
- `scoring`:
  - `percentiles: list[float]` (minimum `[0.5, 0.95]`)
  - `metrics` (paper-aligned):
    - static: `request_execution_plus_preemption_time_normalized`
    - dynamic: `request_e2e_time_normalized`
  - thresholds:
    - `pass_pct: float` (default 0.05)
    - `warn_pct: float` (default 0.09)

**Relationships**

- A `Scenario` produces a `TraceArtifact`.
- A `Scenario` produces a `SimRun`, a `RealRun`, an optional `CapacitySearchRun`, and a `ScoreReport`.

### TraceArtifact

Canonical request stream definition shared by sim + real.

**Files**

- `trace.csv`: `<WORKSPACE_ROOT>/tmp/paper_fidelity/traces/<scenario_name>/trace.csv`
- `trace_meta.json`: `<WORKSPACE_ROOT>/tmp/paper_fidelity/traces/<scenario_name>/trace_meta.json`

**Schema (`trace.csv`)**

- Required:
  - `arrived_at: float` (seconds since start; non-decreasing)
  - `num_prefill_tokens: int` (>= 1)
  - `num_decode_tokens: int` (>= 1)
- Optional:
  - `request_id: int` (unique; defaults to row index)
  - `prompt_id: str` (only when derived from legacy workload dirs)

**Validation**

- `arrived_at >= 0` and non-decreasing
- `num_prefill_tokens >= 1`, `num_decode_tokens >= 1`
- `(num_prefill_tokens + num_decode_tokens) <= max_tokens` (scenario-defined)

### Run (common metadata)

Each run writes a `run_meta.json` capturing reproducibility/provenance.

**Fields**

- `schema_version: str` (e.g., `v1`)
- `run_type: enum {trace, sim, capacity, real, score}`
- `run_id: str` (unique, stable in logs)
- `scenario_name: str`
- `started_at: str` (UTC ISO8601), `ended_at: str` (UTC ISO8601)
- `git_commit: str`, `git_dirty: bool`
- `env`: minimal environment snapshot (Python, torch, CUDA availability, platform)
- `params`: fully resolved Hydra config (or equivalent)
- `artifacts`: absolute paths to produced files

### SimRun

Vidur simulation outputs for a scenario.

**Files**

- `request_metrics.csv`: `<WORKSPACE_ROOT>/tmp/paper_fidelity/runs/<scenario_name>/sim/request_metrics.csv`
- `run_meta.json`: `<WORKSPACE_ROOT>/tmp/paper_fidelity/runs/<scenario_name>/sim/run_meta.json`
- (kept for debugging) raw Vidur metrics dir (e.g., `vidur_raw/.../request_metrics.csv`)

**Required columns (minimum for scoring + capacity)**

- `request_scheduling_delay`
- `request_execution_plus_preemption_time_normalized`
- `request_e2e_time_normalized`
- `request_num_decode_tokens` (or an equivalent “output length” to validate normalization)

### RealRun

Sarathi-Serve replay outputs for a scenario.

**Files**

- `request_metrics.csv`: `<WORKSPACE_ROOT>/tmp/paper_fidelity/runs/<scenario_name>/real/request_metrics.csv`
- `run_meta.json`: `<WORKSPACE_ROOT>/tmp/paper_fidelity/runs/<scenario_name>/real/run_meta.json`

**Required columns (minimum for scoring + capacity)**

- `request_scheduling_delay`
- `request_execution_plus_preemption_time_normalized`
- `request_e2e_time_normalized`
- `request_num_decode_tokens`

### CapacitySearchRun

Search output used to compute the 85% operating point for dynamic workload scoring.

**Files**

- `capacity.json`: `<WORKSPACE_ROOT>/tmp/paper_fidelity/runs/<scenario_name>/capacity/capacity.json`
- `run_meta.json`: `<WORKSPACE_ROOT>/tmp/paper_fidelity/runs/<scenario_name>/capacity/run_meta.json`

**Fields (`capacity.json`)**

- `capacity_qps: float`
- `qps_85: float`
- `criterion`: `{"metric": "request_scheduling_delay", "quantile": 0.99, "threshold_s": 5.0}`

### ScoreReport

Human-readable output summarizing fidelity error.

**Files**

- `summary.md`: `<WORKSPACE_ROOT>/results/reports/<date>/paper_fidelity/<scenario_name>/summary.md`
- Optional: `<WORKSPACE_ROOT>/results/reports/<date>/paper_fidelity/<scenario_name>/tables/*.csv`
- Optional: `<WORKSPACE_ROOT>/results/reports/<date>/paper_fidelity/<scenario_name>/figs/*.png`

**Validation**

- Report contains: scenario definition, commands run, artifact locations, percentile summaries, percent error, and pass/warn/fail thresholds.

## State transitions

The intended lifecycle is:

`planned` → `trace_ready` → `sim_done` → `capacity_done` (dynamic only) → `real_done` → `scored` → `reported`

Failure modes:

- Any step may fail fast with an actionable error (missing model refs, profiling bundles, CUDA unavailable, schema mismatch).
- Partial artifacts are allowed under `<WORKSPACE_ROOT>/tmp/` but must be detected and reported with resume/cleanup guidance.
