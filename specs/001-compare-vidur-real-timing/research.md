# Research Notes: Compare Vidur vs real Qwen3 A100 timing

**Feature**: `001-compare-vidur-real-timing`  
**Date**: 2025-12-30  
**Spec**: `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/spec.md`

## Sources Consulted

- Sarathi usage notes: `/data1/huangzhe/code/gpu-simulate-test/context/summaries/howto-use-sarathi-serve.md`
- Vidur usage notes: `/data1/huangzhe/code/gpu-simulate-test/context/summaries/howto-use-vidur.md`
- Vidur trace replay generator: `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur/vidur/request_generator/trace_replay_request_generator.py`
- Vidur predictor profiling path overrides: `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur/vidur/config/config.py`
- Vidur metrics capabilities (token-level distributions are aggregated): `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur/vidur/metrics/metrics_store.py`

## Decisions

### Decision: Real backend strategy (`real-bench --backend`)

- **Decision**: Implement `real-bench --backend sarathi|transformers`, but treat `transformers` as the Qwen3 ground-truth path initially.
- **Rationale**: The current Sarathi-Serve submodule does not register Qwen3 (`Qwen3ForCausalLM`) per `/data1/huangzhe/code/gpu-simulate-test/context/summaries/howto-use-sarathi-serve.md`. A `transformers` runner can benchmark Qwen3 without patching `extern/`.
- **Alternatives considered**:
  - Patch Sarathi-Serve to add Qwen3 support (higher fidelity scheduler alignment, but increases scope and introduces `extern/` modifications).
  - Use another serving engine (e.g., vLLM) (not currently tracked as a submodule; increases dependency surface).

### Decision: Canonical time representation (workload + metrics)

- **Decision**: Use integer nanoseconds (`*_ns`) relative to run start (monotonic) for all workload schedules and metrics files.
- **Rationale**: Avoids float rounding and clock skew; supports precise joins/validation; easy to convert to seconds where needed.
- **Alternatives considered**:
  - Float seconds (human-readable but introduces precision pitfalls).
  - Epoch timestamps (harder to reproduce across machines; depends on wall clock sync).

### Decision: Canonical workload spec vs Vidur-native input

- **Decision**: Keep the repo’s canonical workload spec as `trace_lengths.csv` + `trace_intervals.csv` (ns), and have `vidur-sim` generate a Vidur trace-replay CSV as an intermediate input.
- **Rationale**: Vidur’s built-in `TRACE_REPLAY` generator expects a single CSV with `arrived_at` (seconds) and token lengths. Generating the intermediate file preserves the “one canonical source of truth” while avoiding changes to Vidur internals.
- **Alternatives considered**:
  - Use Vidur’s `TraceRequestIntervalGenerator` (`arrival_time` datetime-based CSV) (does not match our `*_ns` workload schedule).
  - Patch Vidur to accept our trace formats directly (violates “adapters, not forks” unless justified).

### Decision: Vidur profiling root support

- **Decision**: Implement `vidur-profile` + `vidur-sim` wrappers that accept `--profiling-root` and pass absolute predictor input templates to Vidur (compute/attention/network/cpu overhead) using the `execution_time_predictor_config_*_input_file` knobs.
- **Rationale**: Vidur defaults to `./data/profiling/...` paths, which are CWD-dependent. Overriding these templates allows running from repo root reproducibly.
- **Alternatives considered**:
  - `cd extern/tracked/vidur` before running Vidur (CWD coupling persists; violates the spec’s intent for explicit profiling roots).

### Decision: Token metrics for Vidur runs (`token_metrics.csv`)

- **Decision**: Generate `token_metrics.csv` for Vidur by expanding per-request averages from Vidur’s `request_metrics.csv`, anchored to the workload arrival times:
  - `ttft_ns` derived from Vidur `prefill_e2e_time` (seconds) × 1e9.
  - `per_token_latency_ns` derived from Vidur `decode_time_execution_plus_preemption_normalized` (seconds/token) × 1e9.
  - `token_time_ns[i] = arrival_time_ns + ttft_ns + i * per_token_latency_ns` for `i ∈ [0, num_decode_tokens_requested)`.
- **Rationale**: Vidur’s built-in token-level tracking is stored as aggregated distributions (CDF sketches), not per-request token timestamps. Expanding per-request averages yields a deterministic, comparable proxy in a long-format token file without patching Vidur.
- **Alternatives considered**:
  - Parse Vidur’s internal token-completion distributions/CDFs (loses per-request structure; not “one row per token”).
  - Modify Vidur to export per-token/per-request timestamps (requires changes under `extern/`).

### Decision: Early stopping alignment (real vs sim)

- **Decision**: Allow early stop in real runs; record `num_decode_tokens_actual`; compare per-token metrics by truncating simulated token series to the real token count on a per-request basis.
- **Rationale**: Forcing exact token counts across backends is fragile and can distort “real” behavior. Truncation keeps comparisons meaningful and consistent.
- **Alternatives considered**:
  - Disable EOS/stops and force fixed decode length (not always possible across stacks; changes semantics).
  - Drop early-stopped requests (biases distributions).
