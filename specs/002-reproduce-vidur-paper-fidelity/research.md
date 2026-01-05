# Research: Reproduce Vidur paper fidelity

**Spec**: `<WORKSPACE_ROOT>/specs/002-reproduce-vidur-paper-fidelity/spec.md`  
**Plan**: `<WORKSPACE_ROOT>/specs/002-reproduce-vidur-paper-fidelity/plan.md`  
**Date**: 2026-01-05

This document resolves all `NEEDS CLARIFICATION` items from `plan.md` and records key design decisions.

**Path convention**: `<WORKSPACE_ROOT>` refers to the repository root.

## Real baseline engine (MVP)

**Decision**: Use **Sarathi-Serve** as the required real-backend for the MVP reproduction workflow; treat **vLLM** as an optional follow-up backend.

**Rationale**:

- The feature spec clarifications explicitly require Sarathi-Serve for the MVP.
- This repo already integrates Sarathi as an editable dependency at `<WORKSPACE_ROOT>/extern/tracked/sarathi-serve`.
- Sarathi’s built-in benchmark path supports **token-length-only** workloads via `prompt_token_ids` (no dataset prompts needed), matching the Vidur paper’s “request length characteristics” methodology.
- Sarathi exposes the same metric vocabulary as Vidur (`request_scheduling_delay`, `request_execution_plus_preemption_time_normalized`, etc.) via its metrics store, which is critical for capacity discovery and paper-aligned scoring.

**Alternatives considered**:

- **Implement a vLLM backend immediately**: higher implementation/maintenance cost in this repo; not currently integrated; conflicts with the MVP clarification.
- **Use the existing Transformers backend**: does not exercise scheduling/queueing behavior under load; lacks paper-aligned scheduling-delay semantics.

## Canonical trace input schema + baseline trace source

**Decision**: Standardize on a single canonical `trace.csv` schema (preferred) that is **directly consumable by Vidur** and easily convertible from legacy files:

- Required columns: `arrived_at` (seconds, float), `num_prefill_tokens` (int), `num_decode_tokens` (int)
- Optional columns (ignored by Vidur): `request_id` (int), `prompt_id` (string)

Support legacy split inputs:

- `/.../trace_lengths.csv` with `request_id,prompt_id,num_prefill_tokens,num_decode_tokens`
- `/.../trace_intervals.csv` with `request_id,inter_arrival_ns,arrival_time_ns`

Canonicalization rule: `arrived_at = arrival_time_ns / 1e9`.

Baseline trace source for the default MVP scenario:

- Token-length source: `<WORKSPACE_ROOT>/extern/tracked/vidur/data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv`
  - Provides `num_prefill_tokens` and `num_decode_tokens` distributions derived with the LLaMA2 tokenizer.
- Workload modes:
  - **Static**: set all `arrived_at = 0` (paper “offline” workload).
  - **Dynamic**: generate `arrived_at` from a seeded Poisson arrival process at the chosen QPS (paper “online” workload).

**Rationale**:

- Vidur’s `TraceReplayRequestGenerator` consumes `trace.csv` with `arrived_at,num_prefill_tokens,num_decode_tokens` (see `<WORKSPACE_ROOT>/extern/tracked/vidur/vidur/request_generator/trace_replay_request_generator.py`).
- The repo’s existing legacy workload format already provides equivalent data (arrival times + token lengths) and can be converted deterministically.
- Using token-length-only traces keeps artifacts small and reproducible, while still matching the paper’s fidelity metric definitions (normalized by output length).

**Alternatives considered**:

- **Keep the legacy split files as the canonical trace format**: works for existing commands but adds friction for Vidur (requires a conversion step anyway) and diverges from the feature spec’s preferred `trace.csv`.
- **Store real prompts/text in the trace**: large artifacts; unnecessary for fidelity metrics when both real and sim use identical token counts.
- **Use Sarathi’s interval-trace datetime format**: doesn’t match Vidur’s `arrived_at` trace replay and introduces unnecessary time parsing/filters.

## Capacity discovery metric + overload criterion

**Decision**: Implement capacity discovery as a QPS search using the overload criterion:

- Overloaded if **P99**(`request_scheduling_delay`) **> 5 seconds** (default; configurable per scenario).
- Capacity QPS is the maximum QPS that is *not overloaded*; operating point is `qps_85 = 0.85 * capacity_qps`.

For the real runner, use Sarathi’s per-request metrics output (`sequence_metrics.csv`) which includes `request_scheduling_delay` and the normalized latency fields required for paper plots.

**Rationale**:

- This directly matches the feature spec clarification and the Vidur/Sarathi metric definitions for scheduling delay (`s_r - a_r`).
- Using `request_scheduling_delay` (not TTFT) avoids conflating queueing delay with prefill compute, which would mis-classify heavy-prefill scenarios.

**Alternatives considered**:

- **Use TTFT as “scheduling delay”**: incorrect metric boundary (includes prefill compute).
- **Use throughput-only capacity**: diverges from the paper-style “not overloaded” criterion.

## Performance goal for real-runner instrumentation

**Decision**: Keep measurement overhead bounded by relying on **engine-native telemetry** (Sarathi metrics store) and disabling expensive tracing by default:

- Enable request-level metric output needed for scoring (per-request values).
- Disable optional heavy traces (e.g., Chrome traces, per-token lists) unless explicitly requested by a scenario.
- The wrapper’s incremental overhead should be limited to CSV/JSON I/O under `<WORKSPACE_ROOT>/tmp/paper_fidelity/` and must not add per-token GPU synchronization beyond what the engine requires.

**Rationale**:

- Paper-fidelity metrics require scheduling delay and normalized request-level latencies; Sarathi already computes them.
- Avoiding extra per-token instrumentation reduces distortion of scheduling delay, especially near the overload boundary.

**Alternatives considered**:

- **Client-side token timestamping in Python**: introduces overhead and ambiguous semantics for “schedule time”.
- **Always-on detailed tracing**: useful for diagnosis but too expensive as the default for reproducibility workflows.
