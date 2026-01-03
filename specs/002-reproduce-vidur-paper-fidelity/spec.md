# Feature Specification: Reproduce Vidur paper fidelity

**Feature Branch**: `002-reproduce-vidur-paper-fidelity`  
**Created**: 2026-01-03  
**Status**: Draft  
**Input**: User description: "check what we are going to implement in context/plans/plan-reproduce-vidur-paper-fidelity.md , create at least 6 user stories, and the new branch should be named 002-<what>"

## Reproducibility & Artifacts *(mandatory)*

- **Run command(s)**:
  - `pixi run paper-fidelity repro --scenario <scenario_name> --workload static`
  - `pixi run paper-fidelity repro --scenario <scenario_name> --workload dynamic`
  - `pixi run paper-fidelity score --sim <sim_metrics.csv> --real <real_metrics.csv>`
- **Configs**:
  - `configs/paper_fidelity/` (scenario definitions: model, trace source, arrival process, runtime limits, seeds)
  - `configs/compare_vidur_real/backend/` (real-backend knobs needed for alignment)
- **Outputs**:
  - Trace artifacts: `tmp/paper_fidelity/traces/<scenario_name>/trace.csv` (+ trace metadata)
  - Simulator metrics: `tmp/paper_fidelity/runs/<scenario_name>/sim/request_metrics.csv`
  - Real metrics: `tmp/paper_fidelity/runs/<scenario_name>/real/request_metrics.csv`
  - Scoring + report: `results/reports/<date>/paper_fidelity/<scenario_name>/summary.md` (+ optional plots)
- **Validation**:
  - Automated: scorer unit tests with fixed fixtures under `tests/test_paper_fidelity_scorer.py`
  - Manual: a “known-good” baseline scenario run that produces a complete `summary.md` and required CSVs
- **External assets** (if any):
  - Model references as documented in `models/README.md`
  - Vidur profiling bundles from `extern/tracked/vidur`
  - Real inference engine baseline used for comparison (must support concurrent request injection and server-side timing)
  - Optional secondary engine baseline for cross-checking results (if available)

## User Scenarios & Testing *(mandatory)*

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

### User Story 1 - End-to-end fidelity reproduction report (Priority: P1)

As a contributor validating Vidur fidelity, I want a single documented workflow that runs a baseline scenario end-to-end (trace → simulate → run real → score) so that I can see whether the simulator-vs-real gap matches the paper’s methodology and is within an acceptable error band.

**Why this priority**: This is the core outcome of the initiative; without an end-to-end report, it’s not possible to validate “we are using Vidur correctly.”

**Independent Test**: Run the workflow for a baseline scenario and verify the report (`summary.md`) is produced and includes required static + dynamic error summaries.

**Acceptance Scenarios**:

1. **Given** a baseline scenario definition and required assets are available, **When** I run the reproduction workflow, **Then** I get a report that includes static and dynamic metric summaries for both simulation and real runs.
2. **Given** an existing pair of sim/real metrics files, **When** I run the scoring-only workflow, **Then** I get the same report tables without re-running simulation or real execution.

---

### User Story 2 - Standardized trace generation and validation (Priority: P1)

As a contributor, I want to generate or validate workload traces in a single standardized schema so that both simulation and real execution consume the same request stream definition.

**Why this priority**: Fidelity evaluation is not meaningful if simulation and real runs are driven by different workload representations.

**Independent Test**: Generate a trace from a token-length distribution and confirm schema validity, determinism under a fixed seed, and basic invariants (non-negative token counts; non-decreasing arrivals).

**Acceptance Scenarios**:

1. **Given** a workload definition without explicit arrival times, **When** I generate a trace at a specified request rate and seed, **Then** I get a trace file with required columns and deterministic content for the same inputs.
2. **Given** an existing trace file, **When** I validate it, **Then** the system either confirms it is usable or returns a clear, actionable validation error.

---

### User Story 3 - Simulation run with paper-required metrics (Priority: P1)

As a contributor, I want to run the simulator on a standardized trace while preserving the paper-required normalized metrics so that static and dynamic fidelity scoring can be computed without reinterpretation.

**Why this priority**: If required metrics are missing or transformed inconsistently, fidelity comparisons will be inaccurate and not comparable to the paper.

**Independent Test**: Run simulation for a small trace and verify the output contains the expected normalized metric columns and a run manifest capturing the scenario definition used.

**Acceptance Scenarios**:

1. **Given** a valid trace and a profiling bundle reference, **When** I run simulation, **Then** I get request-level metrics including the normalized latency fields needed for static and dynamic scoring.
2. **Given** missing or incompatible profiling assets, **When** I run simulation, **Then** it fails fast with an error that identifies the missing asset and how to resolve it.

---

### User Story 4 - Real execution replay with aligned timing boundaries (Priority: P2)

As a contributor, I want to replay the same trace against a real inference engine under load and collect request lifecycle timestamps so that I can compute the paper-aligned real metrics and compare them to simulation.

**Why this priority**: Without system-level real timings under concurrent load, the simulator-vs-real gap cannot be evaluated for dynamic workloads.

**Independent Test**: Run a small replay and confirm the output contains the timing fields required to compute queueing delay and end-to-end latency distributions.

**Acceptance Scenarios**:

1. **Given** a valid trace and an engine baseline available on the machine, **When** I run replay, **Then** I get a request-level metrics file with timestamps sufficient to compute scheduling delay and end-to-end time.
2. **Given** the environment does not support running the real baseline (e.g., missing GPU/runtime), **When** I run replay, **Then** it fails with a clear message describing the prerequisite that is missing.

---

### User Story 5 - Capacity discovery for 85% utilization runs (Priority: P2)

As a contributor, I want the system to determine an approximate capacity request rate for a scenario and compute the 85% capacity operating point so that dynamic fidelity runs are executed at the paper’s intended regime.

**Why this priority**: The dynamic fidelity target depends on being near (but below) saturation; a fixed request rate is not transferable across machines or scenarios.

**Independent Test**: Run capacity discovery with bounded search parameters and verify it produces an estimated capacity request rate and the derived 85% operating point, along with the criterion used to determine overload.

**Acceptance Scenarios**:

1. **Given** a scenario definition with a defined overload criterion, **When** I run capacity discovery, **Then** the system outputs an estimated capacity request rate and the derived 85% operating point with a record of measurements taken.
2. **Given** the scenario is overloaded even at the minimum search rate, **When** I run capacity discovery, **Then** it stops and reports that the scenario is not runnable under the current constraints.

---

### User Story 6 - Fidelity scoring and gap diagnosis (Priority: P3)

As a contributor, I want the system to score simulation vs real outputs using the paper’s error methodology and produce a short diagnostic summary when errors exceed a threshold so that I can quickly identify likely causes and next debugging steps.

**Why this priority**: When fidelity is poor, teams need actionable feedback to iterate on metric boundaries, workload alignment, and profiling inputs.

**Independent Test**: Provide two metrics files (sim and real) and verify the scorer produces percentile summaries, percent error, and a diagnosis section when thresholds are exceeded.

**Acceptance Scenarios**:

1. **Given** sim and real metrics files for the same scenario, **When** I run scoring, **Then** I get a summary that reports median/P50 and tail/P95 values and percent error for the static and dynamic metrics.
2. **Given** the percent error exceeds a configurable threshold, **When** I run scoring, **Then** the report includes a “gap diagnosis” section with at least one concrete, testable hypothesis and what evidence triggered it.

### Edge Cases

- Trace contains negative token counts or missing required columns.
- Trace arrivals are not sorted or contain NaN/inf timestamps.
- Dynamic run overloads the system and produces unbounded queueing delay (should be detected and reported).
- Simulation and real runs use mismatched scenario identifiers (should not be scored together without an explicit override).
- Partial results (interrupted run) leave incomplete artifacts (should be detected and reported with resume/cleanup guidance).

## Requirements *(mandatory)*

### Assumptions

- A “baseline scenario” exists that can be run on at least one supported development machine with access to required model assets and a compatible GPU runtime.
- The primary goal is to reproduce methodology and simulator-vs-real error bands for a small, representative slice (not the full-scale paper evaluation).
- Real-vs-sim comparisons are only valid when both sides use the same trace definition and comparable metric boundaries.

### Scope Boundaries (Non-goals)

- Exhaustive reproduction of all paper figures, large-scale multi-node experiments, or full configuration-search pipelines.
- Exact matching of internal forks or non-public traces; the focus is public artifacts and method fidelity.

### Glossary (for readers outside the domain)

- **Request rate**: How many requests are sent per second.
- **Capacity request rate**: The highest request rate where the system still meets a defined “not overloaded” criterion.
- **85% operating point**: A request rate equal to 85% of the capacity request rate.
- **Median / P50**: The value where 50% of requests are faster and 50% are slower.
- **Tail / P95**: The value where 95% of requests are faster and 5% are slower.
- **Percent error**: The relative difference between simulation and real measurements, computed as `abs(sim - real) / real`.

### Functional Requirements

- **FR-001**: System MUST define and validate a single standardized trace schema that both simulation and real execution consume.
- **FR-002**: System MUST support generating dynamic traces from token-length distributions using a deterministic arrival process with a user-specified seed and request rate.
- **FR-003**: System MUST support a static workload mode that represents “all requests arrive at time zero” for the purpose of static fidelity evaluation.
- **FR-004**: System MUST run simulation using a specified profiling bundle source and produce request-level metrics that include the normalized latency fields required for paper-aligned scoring.
- **FR-005**: System MUST run real execution replay driven by the same trace arrivals and produce request-level metrics with timestamps sufficient to compute (a) scheduling delay, (b) execution time excluding scheduling delay, and (c) end-to-end latency.
- **FR-006**: System MUST compute percentile summaries (at minimum median/P50 and tail/P95) for the static and dynamic metrics for both simulation and real outputs.
- **FR-007**: System MUST compute percent error between simulation and real percentile summaries as `abs(sim - real) / real`, reported separately for each metric and percentile.
- **FR-008**: System MUST implement a capacity discovery workflow that outputs (a) an estimated capacity request rate and (b) the derived 85% capacity operating point for a scenario, along with the overload criterion used.
- **FR-009**: System MUST generate a human-readable report that includes: scenario definition, commands run, artifact locations, percentile summaries, percent error, and a pass/fail summary against configurable error thresholds.
- **FR-010**: System MUST support a scoring-only workflow that consumes existing sim/real metrics artifacts and produces the same report tables without re-running simulation or real execution.
- **FR-011**: System MUST store large run artifacts outside tracked source directories (e.g., under `tmp/`), and keep tracked outputs limited to documentation and small summary artifacts.
- **FR-012**: System MUST fail fast with actionable error messages when required external assets (profiling bundles, model assets, real engine prerequisites) are missing.
- **FR-013**: System MUST record enough provenance to allow a different contributor to reproduce the same report on the same machine (scenario ID, seeds, run timestamps, and artifact paths).

### Acceptance Criteria

- **AC-001 (FR-001)**: A trace missing any required column or containing invalid values (negative tokens, unsorted arrivals, NaN/inf timestamps) is rejected with an actionable validation error; a valid trace is accepted.
- **AC-002 (FR-002)**: Given the same token-length inputs, request rate, and seed, trace generation produces identical output across repeated runs.
- **AC-003 (FR-003)**: In static workload mode, all generated requests have arrival time 0 (or the same timestamp) and the trace is still schema-valid.
- **AC-004 (FR-004)**: A simulation run produces a request-level metrics artifact that contains the normalized latency fields required for the static and dynamic fidelity score.
- **AC-005 (FR-005)**: A real replay run produces a request-level metrics artifact with timestamps sufficient to compute scheduling delay, execution time excluding scheduling delay, and end-to-end latency.
- **AC-006 (FR-006)**: The scorer reports median/P50 and tail/P95 summaries for each required metric for both simulation and real runs.
- **AC-007 (FR-007)**: For every reported percentile summary, percent error is computed as `abs(sim - real) / real` and is consistent with a hand-calculated spot check.
- **AC-008 (FR-008)**: Capacity discovery emits a capacity request rate and the derived 85% operating point, and records the overload criterion and measurements used to reach the result.
- **AC-009 (FR-009)**: The generated report contains scenario definition, commands run, artifact locations, percentile summaries, percent error, and a configurable pass/fail threshold evaluation.
- **AC-010 (FR-010)**: Scoring-only mode produces the same report tables given existing metrics artifacts, without requiring any new simulation or real execution.
- **AC-011 (FR-011)**: Large artifacts are written under `tmp/` and reports under `results/`; tracked source directories remain free of large generated artifacts.
- **AC-012 (FR-012)**: Missing external assets produce a failure that clearly names the missing prerequisite and how to provide it.
- **AC-013 (FR-013)**: The report includes enough provenance (scenario identifier, seeds, timestamps, artifact paths) for another contributor to reproduce results on the same machine.

### Key Entities *(include if feature involves data)*

- **Scenario**: A named evaluation configuration (model, workload type, trace source, run limits, seeds, and thresholds).
- **Trace**: A request stream definition with arrival times and token counts used by both simulation and real replay.
- **Simulation Run**: A single execution of the simulator for a scenario and trace, producing request-level metrics plus run provenance.
- **Real Run**: A single execution of the real baseline for a scenario and trace, producing request-level metrics plus run provenance.
- **Request Metric Record**: Per-request timestamps and derived fields required to compute normalized latency distributions and percentiles.
- **Capacity Result**: The discovered capacity request rate and derived 85% operating point for dynamic runs.
- **Fidelity Report**: A human-readable summary containing percentile tables, percent error, thresholds, and artifact pointers.

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: For the baseline scenario, the report includes median/P50 and tail/P95 percentile summaries and percent error for both the static and dynamic paper-aligned metrics.
- **SC-002**: Re-running the same scenario with the same inputs produces identical trace artifacts and scorer outputs (percentiles and percent error) within 0.1% absolute difference.
- **SC-003**: The workflow produces an estimated capacity request rate and the derived 85% operating point for the baseline scenario in a single run without manual tuning beyond providing search bounds.
- **SC-004**: A second contributor can reproduce the baseline report by following the documented commands and prerequisites, producing the same percent error values within 1% absolute difference (allowing for runtime noise).
