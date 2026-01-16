# Feature Specification: Reliable Vidur MLP profiling for driver-launched kernels

**Feature Branch**: `[005-vidur-mlp-cuda-driver]`  
**Created**: 2026-01-16  
**Status**: Draft  
**Input**: User description: "we are going to fix the issue context/issues/known/issue-vidur-mlp-profiling-misses-cuda-driver-kernels.md , based on the plan context/plans/plan-fix-vidur-mlp-profiling-cuda-driver-kernels.md , read them, and new branch should be named 005-<what>"

## Clarifications

### Session 2026-01-16

- Q: When the selected profiling approach fails validation, should the system fall back automatically or fail fast? → A: Fail fast by default; opt-in automatic fallback.
- Q: Should there be a default profiling approach, or must it be selected explicitly each run? → A: No default; explicitly selected via run configuration.
- Q: What should the default validation strictness be? → A: Strict by default; explicit non-strict allowed.
- Q: Where should validation be enforced (staging vs consumption)? → A: Both staging and consumption.
- Q: Which timing targets are “core” for validation? → A: All summary stats for all ops present.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Produce complete MLP timings (Priority: P1)

As a developer generating a profiling root for simulation, I need MLP compute profiling to produce complete, non-missing timing statistics for all operations and summary statistics captured in the MLP timing dataset, even when GPU kernels are launched through the driver path, so simulation does not silently underpredict compute time.

**Why this priority**: Profiling outputs are used to fit per-operation timing predictors. Missing measurements currently get staged as `0.0`, which silently biases simulation fidelity.

**Independent Test**: Run profiling + staging on a workload known to include driver-launched kernels and verify the staged MLP timing dataset contains no missing values for core timing targets.

**Acceptance Scenarios**:

1. **Given** a profiling run where some MLP kernels are launched via the GPU driver path, **When** the profiling output is staged, **Then** each core timing target is present (non-empty) for every input-size row.
2. **Given** a successful staged profiling root, **When** the timing dataset is consumed for training/simulation, **Then** no core timing target value equals `0.0` due to a missing measurement being coerced to zero.

---

### User Story 2 - Fail fast on missing or suspicious data (Priority: P2)

As a developer consuming a profiling root for simulation and reporting, I want the pipeline to detect missing or suspiciously zero-heavy timing targets and stop with a clear message, so bad data does not propagate into simulation results.

**Why this priority**: This prevents silent fidelity regressions and saves debugging time by blocking known-bad inputs early.

**Independent Test**: Provide an input profiling dataset with missing values and confirm staging/validation fails with an actionable error message.

**Acceptance Scenarios**:

1. **Given** a profiling output where any core timing target is missing for any row, **When** staging/validation runs in strict mode, **Then** the run fails and reports which targets are missing and how to remediate.
2. **Given** a profiling output where core timing targets are present but exhibit an unexpectedly high fraction of exact zeros above the “small input” threshold, **When** staging/validation runs, **Then** the run fails or warns (based on configured strictness) and reports the computed zero-rate.
3. **Given** a profiling run fails validation under the selected profiling approach, **When** the user enables opt-in automatic fallback, **Then** the run retries using the fallback approach, completes with no missing core timing targets, and the produced profiling root records which approach was used.
4. **Given** a profiling output with missing core timing targets, **When** staging/validation runs without an explicit strictness setting, **Then** the run fails (strict-by-default) and reports which targets are missing.
5. **Given** a profiling root with missing or suspiciously zero-heavy core timing targets, **When** a consumer loads it for training/simulation/reporting, **Then** validation is applied and either fails or warns according to the configured strictness.

---

### User Story 3 - Prevent regressions (Priority: P3)

As a maintainer, I want automated checks that cover both runtime-launched and driver-launched kernel traces, so future changes do not reintroduce missing timing attribution.

**Why this priority**: It protects long-running profiling pipelines and reduces repeated investigation into silent fidelity drift.

**Independent Test**: Run the unit tests with synthetic traces covering both launch paths and confirm they pass.

**Acceptance Scenarios**:

1. **Given** a synthetic trace where GPU execution is correlated via the runtime launch path, **When** attribution runs, **Then** the computed per-op time is non-zero.
2. **Given** a synthetic trace where GPU execution is correlated via the driver launch path, **When** attribution runs, **Then** the computed per-op time is non-zero.

---

### Edge Cases

- Profiling runs on unsupported hardware or without a functional GPU: profiling should fail clearly rather than produce partial data.
- Very small input sizes where true timings may be ~0: validation should avoid false positives by using a documented “small input” threshold.
- Mixed/partial profiling roots (e.g., interrupted runs): staging should detect missing required files/columns and fail with remediation guidance.
- Existing historical profiling roots that contain missing→zero artifacts: consumers should detect and flag them before training/simulation.

## Requirements *(mandatory)*

### Definitions

- **Input size**: The per-row independent variable for the MLP timing dataset (in this project, token count).
- **Core timing targets**: The validated timing targets: all summary statistics (min/max/mean/median) for all operations present in the MLP timing dataset.
- **Missing value**: An empty/blank cell or null value in the staged timing dataset.
- **Run configuration**: A per-run, declarative configuration used to select the profiling approach, validation strictness, and fallback behavior.
- **Small input threshold (default)**: Input size < 128 tokens.
- **Zero-heavy (default)**: More than 1% of rows with input size ≥ 128 have an exact `0.0` value for a given core timing target.
- **Strict validation (default)**: Validation policy where missing core timing targets or zero-heavy signals cause failure.
- **Non-strict validation**: Validation policy where missing core timing targets still cause failure, but zero-heavy signals produce warnings instead of failure.
- **Staging validation**: Validation applied when producing a profiling root.
- **Consumption validation**: Validation applied when a profiling root is loaded for training, simulation, or reporting.

### Functional Requirements

- **FR-001**: System MUST attribute GPU execution time to profiled MLP operations regardless of whether kernels are launched via the runtime or driver launch path.
- **FR-002**: System MUST produce a staged MLP timing dataset in which all core timing targets are non-missing for successful runs.
- **FR-003**: System MUST NOT silently convert missing timing measurements into `0.0` values for core timing targets.
- **FR-004**: System MUST validate profiling outputs during staging; in strict mode it MUST fail if any core timing target is missing for any row.
- **FR-005**: System MUST detect “suspiciously zero-heavy” core timing targets (per the definition above); in strict mode it MUST fail, and in non-strict mode it MUST warn.
- **FR-006**: System MUST require the profiling approach to be selected explicitly for each profiling run via run configuration; it MUST support at least one alternate (fallback) profiling approach; it MUST fail fast by default on validation failures, and it MUST support an explicit opt-in mode that automatically retries using the fallback approach; it MUST record the final approach used (and whether fallback occurred) in profiling provenance.
- **FR-007**: When validation fails, system MUST report (a) which targets are affected and (b) at least one remediation action a user can take (including rerunning with the fallback approach).
- **FR-008**: System MUST include automated regression tests that demonstrate non-zero attribution for both runtime-launched and driver-launched kernel traces.
- **FR-009**: System MUST run validation in strict mode by default and MUST allow users to explicitly select non-strict validation via run configuration.
- **FR-010**: System MUST apply the same validation rules at both staging and consumption, using the configured strictness.

### Key Entities *(include if feature involves data)*

- **Profiling Run**: A single execution of the profiling pipeline for a given model/device/settings; produces raw profiling outputs.
- **Profiling Root**: A staged, shareable artifact containing the profiling datasets and provenance used by simulation runs.
- **MLP Timing Dataset**: The structured per-input-size timing targets used to train and/or drive simulated compute time for MLP-related operations.
- **Profiling Provenance Record**: Metadata describing how the profiling root was produced (inputs, settings, timestamps, profiling approach).

### Assumptions

- The profiling pipeline supports a “strict” mode suitable for automation.
- A fallback profiling approach exists that can be used when the selected profiling approach cannot produce complete data.
- “Core timing targets” are a defined set of fields used downstream for training/simulation and can be validated consistently.
- The “small input” threshold and “zero-heavy” limit are configurable and have sensible defaults.

### Dependencies

- Access to supported GPU hardware and drivers for profiling the target model(s).
- Availability of correlation metadata that allows mapping GPU execution to profiled operations for both runtime and driver launch paths.
- Downstream consumers that rely on the staged profiling root for training/simulation and reporting.
- A project run-configuration mechanism to select the profiling approach and validation strictness per run.

### Out of Scope

- Improving simulator accuracy beyond eliminating missing/incorrect timing measurements.
- Replacing or redesigning the entire profiling pipeline or report generation stack.
- Retrofitting old profiling roots automatically; the feature focuses on preventing creation/consumption of bad new data.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For a completed profiling run, the staged MLP timing dataset contains 0 missing values in the defined core timing targets across all rows.
- **SC-002**: When strictness is not specified, validation runs in strict mode; the system never produces a usable profiling root with missing core timing targets; if validation cannot be satisfied, the run fails and the error message lists the missing targets.
- **SC-003**: Automated regression tests cover both runtime-launched and driver-launched kernel traces and pass consistently.
- **SC-004**: For a known driver-launch workload, the proportion of exact-zero values in core timing targets above the default “small input” threshold is ≤ 1%.
- **SC-005**: When a profiling root is loaded for training/simulation/reporting, validation detects missing core timing targets with 100% accuracy and blocks usage under strict mode.
