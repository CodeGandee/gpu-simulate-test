# Feature Specification: Vidur CLI Ray runtime configuration (env > config > defaults)

**Feature Branch**: `[006-vidur-cli-ray-config]`  
**Created**: 2026-01-20  
**Status**: Draft  
**Input**: User description: "read about the task we are going to do context/plans/plan-vidur-cli-ray-runtime-config.md , and for new branch of this spec, name it like magic-context/speckit/name-new-branch-by-increment.md"

## Clarifications

### Session 2026-01-20

- Q: Default Ray settings in repo config → A: Opt-in (repo default leaves supported Ray settings unset).
- Q: Object store sizing controls → A: Allow both proportion and max-bytes controls together.
- Q: Initial “supported Ray settings” scope → A: Minimal scope (only the initial 3 settings).
- Q: Unknown Ray settings in config → A: Fail fast (list supported settings).
- Q: Scope of “no-Ray compute profiling” → A: No-Ray means no Ray at all; use fallback outputs for parts that require Ray.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Configure Ray runtime settings from `vidur-cli` config (Priority: P1)

As a `vidur-cli` user running sim-vs-real workflows on a host or in Docker, I can set a small, supported set of Ray runtime settings in the workflow configuration so runs behave consistently across environments without needing to remember manual `RAY_*` exports.

**Why this priority**: Inconsistent Ray defaults across environments can cause avoidable failures (e.g., large object store reservations) and make runs hard to reproduce.

**Independent Test**: Run a Ray-using `vidur-cli` stage with no relevant `RAY_*` environment variables set and with Ray settings provided via config; verify the stage reports the effective settings and completes successfully.

**Acceptance Scenarios**:

1. **Given** the user provides Ray settings via configuration and does not set any corresponding `RAY_*` environment variables, **When** they run a `vidur-cli` stage that uses Ray (e.g., profiling or real replay), **Then** the run uses the configured values and reports them as coming from config.
2. **Given** the user leaves a supported Ray setting unset in configuration, **When** they run a Ray-using stage, **Then** the tool does not inject a value for that setting and reports it as using defaults.
3. **Given** the user uses the same configuration on host and in Docker, **When** they run the same Ray-using stage in both environments, **Then** the tool reports the same effective settings values in both runs.
4. **Given** the user provides an invalid value for a supported Ray setting in configuration, **When** they run a Ray-using stage, **Then** the tool fails fast before starting Ray and reports the invalid setting and how to fix it.
5. **Given** the user provides both object store “max bytes” and “proportion” controls, **When** they run a Ray-using stage, **Then** the tool allows both settings to be applied and reports both as part of the effective settings.
6. **Given** a user wants to avoid known Ray memory spikes in Docker, **When** they consult the workflow documentation, **Then** they can find the supported settings list, the precedence rules, and at least one Docker-friendly example configuration.

---

### User Story 2 - Respect user-provided `RAY_*` environment variables (Priority: P1)

As a power user, if I explicitly set `RAY_*` environment variables, I can rely on `vidur-cli` to respect them and never override them with config defaults.

**Why this priority**: Environment variables are often set intentionally for debugging and operational control; overriding them would be surprising and unsafe.

**Independent Test**: Set a supported `RAY_*` environment variable to a non-default value, run a Ray-using stage with a conflicting config value, and verify the effective settings report shows the environment value is used.

**Acceptance Scenarios**:

1. **Given** a supported `RAY_*` environment variable is set, and the configuration specifies a different value for the same setting, **When** the user runs a Ray-using stage, **Then** the effective value matches the environment variable and the tool reports the source as environment.
2. **Given** multiple supported Ray settings are provided via configuration, **When** some of those settings are also provided via `RAY_*` environment variables, **Then** the tool applies env > config per-setting (mixed sourcing is allowed and visible).
3. **Given** a supported `RAY_*` environment variable is set to an invalid value, **When** the user runs a Ray-using stage, **Then** the tool fails fast and reports which environment variable is invalid and how to fix it.

---

### User Story 3 - Optionally avoid Ray for compute profiling (Priority: P2)

As a user running Vidur compute profiling in a low-footprint environment (especially single-GPU), I can disable Ray usage for compute profiling so profiling can run with lower operational overhead, while keeping the rest of the sim-vs-real workflow compatible.

**Why this priority**: Ray is not always needed for the compute-profiling workload; avoiding it can reduce resource spikes and simplify debugging.

**Independent Test**: Run compute profiling with the “disable Ray for compute profiling” option enabled and verify it completes (for the supported scope) and produces outputs that downstream steps can consume.

**Acceptance Scenarios**:

1. **Given** the user disables Ray for compute profiling and uses a supported single-GPU configuration, **When** they run compute profiling, **Then** it completes without starting Ray and produces compute profiling outputs in the expected format/location.
2. **Given** the user disables Ray for compute profiling but requests an unsupported configuration (e.g., multi-GPU compute profiling), **When** they run compute profiling, **Then** the tool fails fast with an actionable message describing the limitation and how to proceed.
3. **Given** Ray is disabled only for compute profiling, **When** the user runs other workflow steps that still require Ray (e.g., CPU overhead measurement or real replay), **Then** those steps continue to work and still benefit from the same env > config > defaults precedence for Ray runtime settings.
4. **Given** Ray is disabled for compute profiling and some compute profiling work cannot run without Ray, **When** the user runs compute profiling, **Then** the tool skips that work and produces the required fallback outputs for downstream steps, while clearly indicating that fallback outputs were used.

---

### User Story 4 - Fail fast on misconfiguration with actionable errors (Priority: P2)

As a `vidur-cli` user, I get immediate, actionable feedback when Ray runtime settings are misconfigured so I can fix the issue quickly without triggering large resource allocations or long-running steps.

**Why this priority**: Misconfigurations are common during debugging and environment changes; failing fast with clear guidance reduces downtime and prevents confusing failures.

**Independent Test**: Introduce a deliberate configuration or environment mistake for a supported Ray setting and verify the stage exits quickly, without starting Ray, and prints an actionable error.

**Acceptance Scenarios**:

1. **Given** the configuration includes a Ray setting key that is not in the supported settings list, **When** the user runs a Ray-using stage, **Then** the tool fails fast and lists the supported settings.
2. **Given** the configuration provides an invalid value for a supported Ray setting, **When** the user runs a Ray-using stage, **Then** the tool fails fast and reports which setting is invalid and how to correct it.
3. **Given** the environment provides an invalid value for a supported `RAY_*` variable, **When** the user runs a Ray-using stage, **Then** the tool fails fast and reports which environment variable is invalid and how to correct it.

---

### User Story 5 - Configure safely using documentation (Priority: P3)

As a user, I can rely on documentation to configure Ray runtime behavior safely (especially in Docker), understand precedence rules, and know that defaults are opt-in so my runs won’t change behavior unexpectedly.

**Why this priority**: Clear documentation reduces onboarding time and support burden, and it prevents repeated “memory spike” incidents in containerized runs.

**Independent Test**: Locate the documentation section for this feature and verify it includes the supported settings list, precedence rules, the opt-in default behavior, and at least one Docker-friendly example a user can apply.

**Acceptance Scenarios**:

1. **Given** a user is unfamiliar with Ray settings, **When** they read the documentation for this feature, **Then** they can find (a) the supported settings list, (b) precedence rules, (c) that repo defaults leave these settings unset (opt-in), and (d) at least one Docker-friendly example configuration.
2. **Given** the user applies the documented Docker-friendly example and does not set any corresponding `RAY_*` environment variables, **When** they run a Ray-using stage in Docker, **Then** the run uses the configured values and reports them as coming from configuration.

### Edge Cases

- A configured Ray setting value is invalid (wrong type/format or out of allowed range).
- A supported `RAY_*` environment variable is set to an empty string or a non-parseable value.
- The configuration includes a Ray setting that is not in the supported settings list.
- The user configures both an “absolute size” and a “proportion” control for the same resource (ensure both are accepted and clearly reported).
- A run is executed in Docker with different cgroup or shared-memory settings than the host.
- Compute profiling is run with “no-Ray” enabled but downstream steps expect profiling outputs to exist.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Users MUST be able to define the supported set of Ray runtime settings (see “Supported Ray settings”) via the configuration used by `vidur-cli` workflows.
- **FR-002**: The system MUST apply precedence per setting as: **environment > configuration > Ray defaults**.
- **FR-003**: The system MUST NOT override any supported `RAY_*` environment variable that is already set by the user.
- **FR-004**: If a supported setting is omitted in configuration, the system MUST leave it to defaults (i.e., treat it as a no-op).
- **FR-005**: The system MUST apply the effective Ray runtime settings to all `vidur-cli` commands that start Ray as part of the workflow (at minimum: profiling and real replay), so behavior is consistent across stages.
- **FR-006**: The system MUST make the effective settings observable to users by reporting (a) the effective value and (b) whether it came from environment, configuration, or defaults for each supported setting.
- **FR-007**: The system MUST validate supported setting values from both environment and configuration before running a Ray-using stage; invalid values MUST fail fast with an actionable message and MUST NOT start Ray.
- **FR-008**: If the configuration includes an unsupported Ray setting key, the system MUST fail fast and list the supported Ray settings.
- **FR-009**: The system MUST provide an option to disable Ray for Vidur compute profiling (default: enabled/use Ray).
- **FR-010**: When Ray is disabled for compute profiling, the system MUST still produce compute profiling outputs that are compatible with downstream workflow steps; when the user requests an unsupported “no-Ray” configuration, the system MUST fail fast with an actionable message.
- **FR-011**: The system MUST document the supported Ray settings, the precedence rules, that repo defaults leave these settings unset (opt-in), and at least one Docker-friendly example that avoids known memory spikes without requiring manual `export RAY_*` for the common workflow.
- **FR-012**: The system MUST allow object-store sizing controls to be provided together (e.g., both “proportion” and “max bytes”) and MUST NOT treat this as a configuration conflict.
- **FR-013**: The repository’s default workflow configuration MUST leave all supported Ray settings unset, so Ray behavior does not change unless a user opts in via configuration or environment.
- **FR-014**: When Ray is disabled for compute profiling, the system MUST NOT start Ray; if any compute profiling work requires Ray, the system MUST skip that work and instead produce downstream-compatible fallback outputs, and it MUST clearly indicate when fallback outputs were used.

### Supported Ray settings (initial scope)

Only the settings listed below are supported in the initial release.

- **Object store max memory (bytes)**: `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES`
- **Object store memory proportion**: `RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION`
- **Allow object store slow storage**: `RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE`

### Key Entities *(include if feature involves data)*

- **Workflow Configuration**: The user-provided configuration that controls `vidur-cli` behavior for a run (including Ray-related settings and the “no-Ray compute profiling” option).
- **Ray Runtime Setting**: A supported, user-facing runtime knob with an effective value and a source (environment vs configuration vs defaults).
- **Effective Settings Report**: A user-visible summary emitted by `vidur-cli` that lists each supported Ray setting, its effective value, and where it came from.
- **Compute Profiling Outputs**: The artifacts produced by compute profiling that are consumed by later stages in the sim-vs-real workflow.

### Assumptions

- The primary consumer is a developer running sim-vs-real workflows via `vidur-cli` on either host or Docker.
- “No-Ray compute profiling” is initially scoped to the single-GPU case; unsupported cases are rejected explicitly.
- Users may already have ad-hoc `RAY_*` exports in their shell, and they expect those to remain authoritative.

### Dependencies

- Vidur and the real replay backend(s) continue to start Ray internally as part of their current execution model.
- The `vidur-cli` workflow configuration system is able to provide per-run settings and overrides.

### Out of Scope

- Removing Ray usage from workflow steps that require it today (e.g., real replay and CPU overhead measurement).
- Supporting arbitrary Ray settings beyond the “Supported Ray settings (initial scope)” list.
- Supporting multi-node Ray cluster orchestration.
- Automatically detecting and cleaning up leaked Ray processes from previous aborted runs.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: With no relevant `RAY_*` environment variables set, a documented sim-vs-real run (host and Docker) completes successfully using only configuration-defined Ray settings (no manual `export RAY_*` required).
- **SC-002**: When a supported `RAY_*` environment variable is set to a value that conflicts with configuration, the effective settings report shows the environment value is used (0 overrides by configuration).
- **SC-003**: For the same configuration, the effective settings report is identical between host and Docker runs for all supported Ray settings.
- **SC-004**: With “no-Ray compute profiling” enabled in a supported single-GPU configuration, compute profiling completes successfully and produces outputs consumable by downstream stages; with an unsupported configuration, the run fails fast with an actionable error.
- **SC-004**: With “no-Ray compute profiling” enabled in a supported single-GPU configuration, compute profiling completes successfully without starting Ray and produces outputs consumable by downstream stages (using clearly-indicated fallback outputs for any compute profiling parts that cannot run without Ray); with an unsupported configuration, the run fails fast with an actionable error.
- **SC-005**: Automated tests cover the precedence rules (env > config > defaults) and pass for 100% of supported settings and edge cases defined in this specification.
