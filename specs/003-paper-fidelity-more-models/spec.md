# Feature Specification: Paper-fidelity more models

**Feature Branch**: `[003-paper-fidelity-more-models]`  
**Created**: 2026-01-13  
**Status**: Draft  
**Input**: User description: "check the requirement doc in context/tasks/req-test-more-models.md , we are going to implement this, create 6 user stories for that, and new branch should be named 003-<what>"

## Clarifications

### Session 2026-01-13

- Q: What scale should be required for the static+dynamic test matrix runs to count as “done”? → A: Require `--scale small` for both static and dynamic (50 requests).
- Q: For acceptance runs in this feature, should host profiling always include CPU overhead microbenchmarks? → A: Required for all models (always run `profile --include-cpu-overhead`).
- Q: If the machine does not have enough GPUs to meet a model scenario’s required parallelism, what should the test matrix do? → A: Fail fast and write a failure record (blocker category: insufficient GPUs).
- Q: Should the new model scenarios require the paper-aligned parallelism (TP2/TP4), or should we relax it to fit whatever GPUs are available? → A: Configurable, default tp=1, pp=1.
- Q: How should failures be recorded for the “test matrix” procedure? → A: One per-matrix manifest summarizing all runs, including failure details.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Add paper-fidelity scenarios for paper models (Priority: P1)

As a developer, I can select each paper model (excluding Qwen3-0.6B) as a first-class “scenario” and run the paper-fidelity workflow against it.

**Why this priority**: Without scenario definitions, there is no supported path to run profiling/repro for additional models.

**Independent Test**: A developer can select each new scenario and generate a valid canonical trace for both static and dynamic workloads.

**Acceptance Scenarios**:

1. **Given** the repo is set up and the developer selects the InternLM-20B scenario, **When** they generate a canonical trace for static and dynamic workloads, **Then** a valid trace artifact exists and can be reused as input for later steps.
2. **Given** the repo is set up and the developer selects the LLaMA2-70B or Qwen-72B scenario, **When** they generate a canonical trace for static and dynamic workloads, **Then** the trace schema is valid and consistent with the paper-fidelity workflow.

---

### User Story 2 - Generate host-matched profiling roots per model (Priority: P1)

As a developer, I can generate a host-matched profiling bundle for each in-scope paper model so simulator outputs are meaningfully comparable to real replay on the same machine.

**Why this priority**: Host profiling is required for meaningful sim-vs-real percent error interpretation; paper-provided profiling is not sufficient for host-fidelity conclusions.

**Independent Test**: For each model scenario, host profiling completes and produces a Vidur-compatible profiling root with the expected structure.

**Acceptance Scenarios**:

1. **Given** the model’s assets are available locally and sufficient GPUs are available, **When** the developer runs host profiling for that model (including CPU overhead microbenchmarks), **Then** a profiling root is produced and is usable as an input to simulation.
2. **Given** a host profiling attempt fails (e.g., missing assets or insufficient GPUs), **When** the run ends, **Then** the developer can see a clear, recorded reason and what was attempted.

---

### User Story 3 - Produce a static paper-fidelity report for each model (Priority: P2)

As a developer, I can run a static paper-fidelity reproduction per model and get a scored report that compares simulator outputs vs real replay.

**Why this priority**: Static runs are the simplest end-to-end pipeline and provide a baseline for validating model support and parity-critical settings.

**Independent Test**: A static run per model produces a report directory containing a human-readable summary and machine-readable scoring outputs.

**Acceptance Scenarios**:

1. **Given** a host profiling root exists for a model, **When** the developer runs the static paper-fidelity reproduction at `--scale small` (50 requests), **Then** a report exists with summary + score outputs and the input CSVs used to generate it.

---

### User Story 4 - Produce a dynamic paper-fidelity report for each model (Priority: P2)

As a developer, I can run a dynamic paper-fidelity reproduction per model and get a scored report, including capacity discovery and a timed trace.

**Why this priority**: Dynamic runs validate the capacity-search + timed-arrivals path and are required for the dynamic fidelity metrics pipeline.

**Independent Test**: A dynamic run per model produces a report directory and includes trace and capacity artifacts needed for debugging.

**Acceptance Scenarios**:

1. **Given** a host profiling root exists for a model, **When** the developer runs the dynamic paper-fidelity reproduction at `--scale small` (50 requests), **Then** a report exists that includes a timed trace snapshot and capacity outputs.

---

### User Story 5 - Run the full model/workload matrix with one repeatable procedure (Priority: P3)

As a developer, I can run a repeatable “test matrix” procedure that executes the required model set and workloads, and I can easily find the outputs for each run.

**Why this priority**: Manually running multiple commands per model is error-prone and makes it hard to compare outcomes across models.

**Independent Test**: A single documented procedure produces a per-model/per-workload set of outputs and a clear summary of where the reports are located.

**Acceptance Scenarios**:

1. **Given** the developer has the required model assets and hardware, **When** they run the test matrix procedure (static+dynamic at `--scale small`), **Then** the procedure produces a discoverable output for each model/workload combination and a summary of what succeeded.
2. **Given** the configured parallelism for a model cannot be satisfied by the available GPUs, **When** the developer runs the test matrix procedure, **Then** it fails fast and writes a failure record identifying “insufficient GPUs” as the blocker category.

---

### User Story 6 - Diagnose and record failures consistently (Priority: P3)

As a developer, when a model cannot complete due to resource limits or unsupported configurations, I get a structured failure record that lets me reproduce and categorize the blocker.

**Why this priority**: These runs are expensive; failures must be debuggable without rerunning blindly.

**Independent Test**: Intentionally triggering a failure produces a failure record containing the attempted action, the error message, and the blocker category.

**Acceptance Scenarios**:

1. **Given** a model run fails for a known reason (missing model assets, insufficient GPUs, unsupported model), **When** the run ends, **Then** a failure record is written capturing what was attempted and how to classify the blocker.
2. **Given** a run fails due to insufficient GPUs, **When** the failure record is written, **Then** the blocker category is recorded as “insufficient GPUs”.

---

### Edge Cases

- Insufficient GPU capacity for the configured parallelism (e.g., tensor parallel size > available GPUs). The run must fail fast and write a failure record (blocker category: insufficient GPUs).
- Missing or invalid model asset reference (local model directory cannot be loaded by the real replay runner).
- The real replay runner cannot load the selected model architecture.
- Host profiling completes but produces incomplete or invalid profiling inputs (e.g., missing required profiling outputs).
- Dynamic capacity discovery fails to find a stable operating point within bounded attempts.
- Runs fail due to out-of-memory or timeouts; results must still be recorded as actionable failures.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support running the paper-fidelity workflow for the in-scope paper models: InternLM-20B, LLaMA2-70B, and Qwen-72B.
- **FR-002**: System MUST explicitly exclude Qwen3-0.6B from the “paper model” test matrix.
- **FR-003**: System MUST allow a developer to generate a canonical paper-fidelity trace for both static and dynamic workloads for each in-scope model.
- **FR-004**: System MUST allow a developer to generate a host-matched profiling root for each in-scope model, including CPU overhead microbenchmarks.
- **FR-005**: System MUST allow a developer to run a bounded static reproduction per model at `--scale small` (50 requests) that produces a scored report comparing simulator outputs vs real replay.
- **FR-006**: System MUST allow a developer to run a bounded dynamic reproduction per model at `--scale small` (50 requests) that produces a scored report and includes capacity discovery outputs and a timed trace.
- **FR-007**: System MUST make report outputs discoverable and self-contained by including the inputs used to generate the report (so results are portable and not dependent on mutable temporary directories).
- **FR-008**: System MUST provide a repeatable procedure to execute the required model/workload matrix and identify where the outputs for each run are located.
- **FR-009**: When any required run fails, System MUST record (a) what was attempted, (b) the error message, and (c) a blocker category sufficient for triage (including “insufficient GPUs”).
- **FR-010**: System MUST allow configuring model parallelism per scenario (e.g., tensor/pipeline parallel sizes), with defaults of tp=1 and pp=1.
- **FR-011**: System MUST write a per-matrix manifest that summarizes all attempted runs and includes failure details for unsuccessful runs.

### Dependencies & Assumptions

- The developer running the workflow has access to a CUDA-capable machine appropriate for the model’s required parallelism.
- The model assets referenced by each scenario are present locally and are loadable by the real replay runner.
- This feature validates that the pipeline runs end-to-end and produces artifacts; it does not require the resulting sim-vs-real percent error to match the Vidur paper.

### Key Entities *(include if feature involves data)*

- **Scenario**: A named configuration that defines a model, trace source, and parity-critical settings for simulator and real replay.
- **Model Under Test**: A specific paper model instance (InternLM-20B, LLaMA2-70B, Qwen-72B) with a machine-local asset reference.
- **Profiling Root**: A host-generated profiling bundle used as simulator input for that model on that machine.
- **Test Run**: One execution of (profile, static repro, or dynamic repro) for a specific model/workload/scale.
- **Failure Record**: A structured record capturing a failed attempt and its blocker category.
- **Report**: A scored output bundle that includes a human-readable summary plus machine-readable scoring outputs and input snapshots.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For each in-scope model, at least one host profiling root is produced and at least one static and one dynamic paper-fidelity report are successfully generated at `--scale small` (50 requests).
- **SC-002**: Each successful static run produces a report bundle that includes (at minimum) a human-readable summary, machine-readable scores, and snapshots of the inputs used for scoring.
- **SC-003**: Each successful dynamic run produces a report bundle that includes (at minimum) a timed trace snapshot and capacity discovery outputs in addition to the standard report artifacts.
- **SC-004**: For any failed run, a developer can find a failure record that includes the attempted action, the error message, and a blocker category, without rerunning the workload.
