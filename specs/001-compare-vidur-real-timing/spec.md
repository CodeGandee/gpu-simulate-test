# Feature Specification: Compare Vidur vs real Qwen3 A100 timing

**Feature Branch**: `001-compare-vidur-real-timing`  
**Created**: 2025-12-30  
**Status**: Draft  
**Input**: User description: "read context/plans/plan-compare-vidur-vs-real-qwen3-a100-timing.md first, that is what we want to do, create at least 6 use cases"

## Clarifications

### Session 2025-12-30

- Q: Which stack should `pixi run real-bench` use for ground-truth timing? → A: Support both backends (`sarathi`, `transformers`) via `--backend`.
- Q: How should per-token timing be persisted for real + sim runs? → A: `request_metrics.csv` + `token_metrics.csv` (long format, one row per token).
- Q: What should be the canonical time representation for `trace_intervals.csv` and all timestamp fields in metrics files? → A: Integer nanoseconds, relative to run start (monotonic).
- Q: How should the comparison handle early stopping (real run produces fewer decode tokens than requested)? → A: Allow early stop; record `num_decode_tokens_actual`; truncate sim series to actual for per-token comparisons.
- Q: What should `trace_intervals.csv` contain? → A: Both `arrival_time_ns` and `inter_arrival_ns` (consistent by construction).

## Reproducibility & Artifacts *(mandatory)*

- **Run command(s)**:
  - `pixi install`
  - `pixi run workload-spec --model Qwen/Qwen3-0.6B --prompts <prompts.jsonl> --out tmp/workloads/<workload_id>`
  - `pixi run real-bench --backend <sarathi|transformers> --workload tmp/workloads/<workload_id> --out tmp/real_runs/<run_id>`
  - `pixi run vidur-profile --model Qwen/Qwen3-0.6B --profiling-root <local_profiling_root>`
  - `pixi run vidur-sim --workload tmp/workloads/<workload_id> --profiling-root <local_profiling_root> --out tmp/vidur_runs/<run_id>`
  - `pixi run compare-runs --real tmp/real_runs/<run_id> --sim tmp/vidur_runs/<run_id> --out tmp/comparisons/<run_id>`
- **Configs**:
  - Workload spec inputs/outputs: `tmp/workloads/<workload_id>/` including `prompts.jsonl`, `trace_lengths.csv`, `trace_intervals.csv` (`arrival_time_ns`, `inter_arrival_ns`), and `workload_meta.json`.
  - Comparison run metadata: `tmp/real_runs/<run_id>/run_meta.json`, `tmp/vidur_runs/<run_id>/run_meta.json` (hardware/model knobs and provenance).
- **Outputs**:
  - **Workload**: `tmp/workloads/<workload_id>/` (small text artifacts; can be copied into `context/` if a workload needs to be versioned).
  - **Real**: `tmp/real_runs/<run_id>/request_metrics.csv`, `tmp/real_runs/<run_id>/token_metrics.csv`, and a short `summary.md`.
  - **Vidur**: `tmp/vidur_runs/<run_id>/request_metrics.csv`, `tmp/vidur_runs/<run_id>/token_metrics.csv` (+ plots/trace if enabled), and a short `summary.md`.
  - **Comparison**: `tmp/comparisons/<run_id>/summary.md` plus percentile tables and CDF/percentile plots.
- **Validation**:
  - Manual: `tests/manual/` (end-to-end smoke run with tiny workload, writes to `tmp/`).
  - Unit: `tests/unit/` (validates required columns and determinism for trace files).
- **External assets** (if any):
  - Qwen3 tokenizer/config reference: `models/qwen3-0.6b/source-data/` (external reference pattern).
  - A100 machine access for the real timing run; Vidur profiling data stored locally under `tmp/vidur_profiling/` (not committed).

## User Scenarios & Testing *(mandatory)*

Definitions used in this spec: TTFT is the time from request arrival to first token; per-token latency is the time between consecutive tokens during decode; CDF/percentile plots are distribution plots that summarize latency across many requests/tokens.

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently

  Per the constitution: every change MUST include at least one of manual, unit, or
  integration validation under `tests/`. Prefer manual scripts for experiment-heavy flows.
-->

### User Story 1 - Generate deterministic workload spec (Priority: P1)

As a contributor, I can generate a deterministic workload spec (request token lengths + exact arrival schedule) from a prompt set so that both the real benchmark and the simulator are driven by identical inputs.

**Why this priority**: Without a shared workload definition, “real vs sim” comparisons are not apples-to-apples and are hard to reproduce.

**Independent Test**: Generate a small workload spec from 3 prompts and verify the trace files exist, have the required columns, and are identical across two runs with the same seed.

**Acceptance Scenarios**:

1. **Given** a prompts JSONL file and a fixed seed, **When** I generate a workload spec, **Then** `trace_lengths.csv` contains `num_prefill_tokens` and `num_decode_tokens` for every prompt.
2. **Given** the same prompts and seed, **When** I generate the workload spec twice, **Then** the resulting `trace_intervals.csv` is identical (bit-for-bit) across both runs.
3. **Given** a generated workload spec, **When** I inspect `trace_intervals.csv`, **Then** it contains `arrival_time_ns` and `inter_arrival_ns` as integer nanoseconds relative to workload start, and `arrival_time_ns` is consistent with the cumulative sum of `inter_arrival_ns`.

---

### User Story 2 - Collect real A100 timing metrics (Priority: P1)

As a contributor, I can run a real A100 timing benchmark driven by the workload spec and get a standardized metrics file with per-request TTFT and per-token latency measurements.

**Why this priority**: We need ground-truth timing distributions to evaluate simulator fidelity and to spot where the simulator diverges.

**Independent Test**: Run a tiny workload (e.g., 2–5 requests) and verify `request_metrics.csv` + `token_metrics.csv` contain required columns and plausible timestamp ordering.

**Acceptance Scenarios**:

1. **Given** a workload spec and access to an A100, **When** I run the real benchmark, **Then** it writes `tmp/real_runs/<run_id>/request_metrics.csv` with one row per request.
2. **Given** `request_metrics.csv` and `token_metrics.csv`, **When** I compute TTFT and per-token latencies, **Then** TTFT is non-negative and per-token latencies exist for every generated token that was produced.
3. **Given** `--backend sarathi` and `--backend transformers`, **When** I run `real-bench`, **Then** both runs produce the same `request_metrics.csv` schema and record the chosen backend in `run_meta.json`.

---

### User Story 3 - Run Vidur simulation from repo root (Priority: P1)

As a contributor, I can run a Vidur simulation from the repo root using the same workload spec and produce a standardized metrics file comparable to the real benchmark output.

**Why this priority**: A reproducible simulator run is necessary to compare predicted timing distributions to real measurements and iterate on alignment.

**Independent Test**: Run the simulator on a tiny workload with a known profiling root and verify the run produces `request_metrics.csv` + `token_metrics.csv` and basic summary artifacts under `tmp/`.

**Acceptance Scenarios**:

1. **Given** a workload spec and local profiling data, **When** I run the simulator from repo root, **Then** it writes `tmp/vidur_runs/<run_id>/request_metrics.csv`, `tmp/vidur_runs/<run_id>/token_metrics.csv`, and a `run_meta.json` documenting inputs.
2. **Given** missing profiling data, **When** I attempt to run the simulator, **Then** the run fails fast with an actionable message describing what profiling inputs are required.

---

### User Story 4 - Produce a comparison report (Priority: P1)

As a contributor, I can generate a comparison report between one real run and one simulator run that includes percentile tables and distribution plots for TTFT and per-token latency.

**Why this priority**: A single, standardized report format makes it easy to review fidelity gaps, share results, and iterate on simulator/benchmark alignment.

**Independent Test**: Using two existing run directories (one real, one simulated), generate a report and verify it contains percentiles and at least one plot per metric family.

**Acceptance Scenarios**:

1. **Given** a real run directory and a simulator run directory, **When** I run the comparison, **Then** it writes `tmp/comparisons/<run_id>/summary.md` with P50/P90/P99 for TTFT and per-token latency.
2. **Given** missing or malformed metrics inputs, **When** I run the comparison, **Then** it fails with a clear error pointing to the missing columns/files.
3. **Given** early-stopped requests (fewer tokens than requested), **When** I run the comparison, **Then** per-token comparisons use `num_decode_tokens_actual` to align token series (truncate sim tokens to the real token count for those requests).

---

### User Story 5 - Generate Vidur profiling bundle (Priority: P2)

As a contributor, I can generate and stage Vidur profiling data for Qwen/Qwen3-0.6B on an A100 under a profiling root so the simulator can run reproducibly.

**Why this priority**: Vidur simulation depends on model+GPU-specific profiling inputs; generating them into an explicit profiling root avoids hidden path dependencies.

**Independent Test**: Run profiling for Qwen/Qwen3-0.6B on A100 and verify the profiling root contains the required CSV files and metadata.

**Acceptance Scenarios**:

1. **Given** access to an A100 and a writable profiling root, **When** I run profiling, **Then** it writes the required profiling CSV inputs under `<local_profiling_root>` and records the model + hardware identifiers used.
2. **Given** missing GPU access or an invalid model reference, **When** I run profiling, **Then** it fails fast with an actionable message.

---

### User Story 6 - Run Vidur without patching the submodule (Priority: P2)

As a contributor, I can run Vidur for `Qwen/Qwen3-0.6B` from repo root without modifying `extern/tracked/vidur`, so submodule upgrades do not require reapplying local patches.

**Why this priority**: Keeping integrations in-repo (not in the Vidur submodule) reduces rebase/merge friction and makes provenance clearer.

**Independent Test**: Run a Vidur entrypoint that references `Qwen/Qwen3-0.6B` and verify it starts without model-registration/import errors.

**Acceptance Scenarios**:

1. **Given** a clean checkout, **When** I run `vidur-profile` or `vidur-sim` for `Qwen/Qwen3-0.6B`, **Then** Vidur recognizes the model identifier without any changes under `extern/tracked/vidur/`.
2. **Given** the workflow is set up, **When** I check `git status`, **Then** no modifications under `extern/tracked/vidur/` are required to run Qwen3 profiling or simulation.

### Edge Cases

- Missing prerequisites: no A100 available, missing model reference, or missing simulator profiling data.
- Trace/schema mismatch: required columns missing, non-deterministic arrivals, or tokenization mismatch causing real and sim to use different lengths.
- Early stopping: real runs may end early (EOS/stop sequences) and produce fewer tokens than requested; record `num_decode_tokens_actual` and compare per-token latencies only for produced tokens (truncate sim token series to actual).
- Cold start vs steady state: warmup requests distort distributions; reports must support excluding warmup windows or labeling them.
- Partial artifacts: interrupted runs leave incomplete directories; tooling must detect and refuse to compare incomplete runs.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST define a canonical workload spec format that can drive both real and simulated runs.
- **FR-002**: System MUST generate `trace_lengths.csv` and `trace_intervals.csv` deterministically from the same inputs (prompts + seed + configuration); `trace_intervals.csv` MUST include `arrival_time_ns` and `inter_arrival_ns` as integer nanoseconds relative to workload start.
- **FR-003**: System MUST run a real timing benchmark on an A100 using the workload spec and emit standardized `request_metrics.csv` (per-request) + `token_metrics.csv` (per-token, long format); `real-bench` MUST support `--backend sarathi|transformers`.
- **FR-004**: System MUST run a Vidur simulation from repo root using the same workload spec and emit standardized `request_metrics.csv` (per-request) + `token_metrics.csv` (per-token, long format) with comparable TTFT/per-token timing proxies.
- **FR-005**: System MUST support an explicit, user-provided profiling root so the simulator run does not depend on current working directory or submodule-relative paths.
- **FR-006**: System MUST generate a comparison report that includes percentile tables (P50/P90/P99) and plots for TTFT and per-token latency, for a single real+sim pair; comparisons MUST handle early stopping by aligning per-token metrics on `num_decode_tokens_actual` (truncate sim token series to actual).
- **FR-007**: System MUST write all generated artifacts under `tmp/` by default and MUST avoid committing large generated outputs to the repository.
- **FR-008**: System MUST record run metadata sufficient to reproduce results: model identifier, hardware identifier, workload spec reference, and timestamps.
- **FR-009**: System MUST fail fast with actionable messages when prerequisites are missing (GPU access, model reference, profiling data, or required trace columns).

### Requirement Acceptance Mapping

- **FR-001–FR-002**: Covered by User Story 1 acceptance scenarios (trace schema + determinism).
- **FR-003**: Covered by User Story 2 acceptance scenarios (standardized metrics output + backend support + TTFT/per-token plausibility).
- **FR-004–FR-005**: Covered by User Story 3 acceptance scenarios (repo-root execution + profiling-root failures).
- **FR-006**: Covered by User Story 4 acceptance scenarios (report content + input validation).
- **FR-007–FR-009**: Covered across User Stories 1–6 and Edge Cases (artifact location, metadata, and fast-fail behavior).

### Scope

In scope:
- Qwen/Qwen3-0.6B inference timing comparisons on A100 with a shared, deterministic workload spec.
- Metrics focused on TTFT and per-token latency distributions, with standardized artifacts and a comparison report.

Out of scope (initially):
- Multi-GPU (TP/PP > 1) topologies and network profiling.
- Training workloads or fine-tuning performance.
- A general “any model/any GPU” framework beyond what is needed for Qwen3-0.6B on A100.

### Dependencies & Assumptions

- An A100 machine is available for the real benchmark measurements.
- The Qwen3 model reference (tokenizer/config) is available via the repo’s external reference pattern under `models/`.
- A local profiling root for the simulator exists or can be generated on a compatible GPU; it is treated as machine-local data under `tmp/` and is not committed.
- Real benchmark measurements can record request arrival time, first-token time, and per-token timestamps (persisted to `token_metrics.csv`) needed to derive TTFT and per-token latency.
- All timing fields in `trace_intervals.csv`, `request_metrics.csv`, and `token_metrics.csv` use integer nanoseconds relative to run start (monotonic).

### Key Entities *(include if feature involves data)*

- **WorkloadSpec**: A deterministic definition of the request stream (prompts + token-length trace + arrival schedule) with an identifier and provenance.
- **TraceLengths**: Per-request token counts (`num_prefill_tokens`, `num_decode_tokens`) derived from the chosen tokenizer and prompts.
- **TraceIntervals**: Arrival schedule defining when each request is issued (`arrival_time_ns`, `inter_arrival_ns`).
- **RunMetadata**: Captures run context (hardware/model identifiers, workload reference, timestamps, and key run parameters) to enable reproducibility.
- **RequestMetrics**: Per-request measured or simulated timing signals including TTFT and `num_decode_tokens_actual` (per-request summary rows).
- **TokenMetrics**: Per-token timestamp or delta series in long format (one row per generated token) sufficient to derive per-token latency distributions.
- **ComparisonReport**: A human-readable summary with percentiles, plots, and links to the underlying artifacts for both runs.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A contributor can produce one end-to-end baseline comparison (workload → real run → sim run → report) using only documented steps, and all expected artifacts are created under `tmp/`.
- **SC-002**: For a fixed prompt set and seed, regenerating the workload spec produces identical trace files (determinism verified via file hashes or byte-for-byte comparison).
- **SC-003**: The comparison report includes P50/P90/P99 tables and CDF/percentile plots for both TTFT and per-token latency, for both real and simulated runs.
- **SC-004**: At least 6 use cases are documented as independently testable user stories, each with acceptance scenarios that can be verified from produced artifacts.
- **SC-005**: Every run directory includes machine-readable metadata that identifies the model, hardware, workload, and time window used for measurement.
