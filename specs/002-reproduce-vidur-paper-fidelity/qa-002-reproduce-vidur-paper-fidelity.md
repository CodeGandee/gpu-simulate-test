# Q&A: 002-reproduce-vidur-paper-fidelity

## Introduction

This Q&A captures practical clarifications that come up when implementing and validating the Vidur paper-fidelity reproduction workflow, for developers (including future maintainers).

**Related docs**
- `specs/002-reproduce-vidur-paper-fidelity/spec.md`
- `specs/002-reproduce-vidur-paper-fidelity/plan.md`
- `specs/002-reproduce-vidur-paper-fidelity/research.md`
- `specs/002-reproduce-vidur-paper-fidelity/data-model.md`
- `specs/002-reproduce-vidur-paper-fidelity/quickstart.md`
- `specs/002-reproduce-vidur-paper-fidelity/contracts/paper_fidelity.openapi.yaml`

**Key entrypoints and modules**
- `extern/tracked/vidur/docs/metrics.md`
- `extern/tracked/vidur/paper/tex/5-eval.tex`
- `extern/tracked/vidur/paper/tex/figures-tex/fig-fidelity-static-trace.tex`
- `extern/tracked/vidur/paper/tex/figures-tex/fig-fidelity-dynamic-trace.tex`
- `extern/tracked/sarathi-serve/sarathi/metrics/README.md`
- `src/gpu_simulate_test/vidur_ext/sim_runner.py`
- `src/gpu_simulate_test/real_bench/backends/sarathi_backend.py`

## What does Vidur actually measure in its timing metrics?
> Last revised at: `2026-01-05T03:27:58Z` | Last revised base commit: `b735cecbb66fde6f6f956c68ec34f59877e0725a`

- Vidur reports *request lifecycle timings inside the serving system boundary* (not client/network time), using the canonical request timestamps defined in `extern/tracked/vidur/docs/metrics.md`:
  - arrival time `a_r`, first schedule time `s_r`, prefill completion time `f_r` (first output token), completion time `c_r`
  - derived times like scheduling delay `d_r = s_r - a_r` and end-to-end latency `c_r - a_r`
- It also decomposes latency into modeled “work” vs “waiting”:
  - execution time `e_r` (GPU execution time across attempts)
  - preemption / non-executing time `p_r` (pipeline bubbles, stalls, restarts, etc.)
  - execution-plus-preemption time `(c_r - s_r)` which is the time “in system excluding initial scheduling delay”
- In `request_metrics.csv`, Vidur emits these as per-request distributions (seconds) and normalized variants (seconds per output token), e.g.:
  - `request_e2e_time`, `request_scheduling_delay`
  - `request_e2e_time_normalized`
  - `request_execution_plus_preemption_time_normalized`
  - example header: `tmp/compare_experiments/.../vidur_raw/.../request_metrics.csv`
- The Vidur paper’s fidelity figures explicitly use:
  - static: `request_execution_plus_preemption_time_normalized` at P50/P95 (`extern/tracked/vidur/paper/tex/figures-tex/fig-fidelity-static-trace.tex`)
  - dynamic: `request_e2e_time_normalized` at 85% capacity at P50/P95 (`extern/tracked/vidur/paper/tex/figures-tex/fig-fidelity-dynamic-trace.tex`)

## How do Vidur’s timing metrics differ from end-to-end LLM inference, and what may be missing?
> Last revised at: `2026-01-05T03:27:58Z` | Last revised base commit: `b735cecbb66fde6f6f956c68ec34f59877e0725a`

- Vidur’s “end-to-end” (`c_r - a_r`) starts at *system ingress* (`a_r`) and ends at *system completion* (`c_r`); it does not include:
  - client → server network latency, load balancer/proxy time, HTTP/gRPC overheads
  - request parsing, prompt tokenization, response detokenization/serialization
  - client-side concurrency effects (connection pools, retries, timeouts)
- Vidur’s timings are **simulated** from profiled/predicted operator runtimes + modeled scheduling, so it may miss or smooth over:
  - CPU-side scheduling/dispatch overheads (especially impactful for small models)
  - OS/runtime jitter and contention (Python interpreter overhead, thread scheduling)
  - kernel-level variability, launch overheads, allocator behavior, and driver/runtime edge cases
- The Vidur paper calls out CPU overhead as a real-vs-sim gap contributor for smaller models (see the discussion in `extern/tracked/vidur/paper/tex/5-eval.tex`).
- In this repo’s current Vidur wrapper (`src/gpu_simulate_test/vidur_ext/sim_runner.py`), CPU overhead modeling is explicitly skipped (`skip_cpu_overhead_modeling=True`), which makes “what Vidur measures” closer to “GPU + modeled scheduler time” than full system time.

## In this repo’s current Vidur integration, do we profile microbenchmarks on our GPU, and can partial profiling create a large sim-vs-real gap?
> Last revised at: `2026-01-05T04:51:53Z` | Last revised base commit: `b735cecbb66fde6f6f956c68ec34f59877e0725a`

- To simulate a target GPU/topology faithfully, Vidur needs profiling CSVs measured for that GPU + kernel/backend choices; once you have those CSVs, the *simulation* can run anywhere, but the *profiling* must run on that target hardware. You either (a) use Vidur’s shipped “default profiles” for that hardware, or (b) run Vidur’s profiling on that same hardware to generate new CSVs.
- In this repo, `pixi run vidur-profile` runs GPU microbenchmarks locally (requires CUDA) and writes compute profiling under `tmp/vidur_profiling/<hardware>/<model_key>/data/profiling/compute/...` (see `src/gpu_simulate_test/cli/vidur_profile.py` → `src/gpu_simulate_test/vidur_ext/profile_runner.py`), with provenance in `tmp/vidur_profiling/<hardware>/<model_key>/run_meta.json`.
- Our current attention profiling invocation is intentionally minimal (`TP=1`, `batch=1`, `--profile_only_decode`), which means it does **not** cover the full set of attention cases Vidur models (Vidur trains separate `attn_prefill` and `attn_decode` predictors). This makes it easy to end up with partial or fallback attention data.
- Network profiling is **not** microbenchmarked locally in this workflow; we stage `all_reduce.csv` / `send_recv.csv` by copying Vidur’s shipped network profiles from `extern/tracked/vidur/data/profiling/network/<network_device>/...`. The key requirement is choosing a `network_device` that matches the real machine’s topology.
- Vidur’s shipped compute profiles (when available for the hardware/model) are more “complete” than our minimal local sweep: they include model-specific `mlp.csv` + `attention.csv` generated by Vidur’s profiler across a wider range of sizes, and include both prefill and decode attention rows needed for training.
- CPU overhead modeling is disabled by default (both in Vidur and in our wrapper): we run with `skip_cpu_overhead_modeling=True`, and the Vidur profiling bundle vendored in this repo does not include `cpu_overheads.csv`. That host/runtime overhead can dominate batch=1 latency and is a common source of underprediction.
- Net effect: partial profiling (local compute + shipped network + possibly templated attention) can create a large sim-vs-real gap. For fidelity-critical workflows, prefer paper-aligned shipped profiles, and avoid “silent” fallbacks by failing fast or recording fallback provenance explicitly.

## In the research paper, how do the authors make Vidur’s measurement and Sarathi-Serve measurement comparable without too much overhead?
> Last revised at: `2026-01-05T03:27:58Z` | Last revised base commit: `b735cecbb66fde6f6f956c68ec34f59877e0725a`

- They align **workloads** and **metric definitions** so the numerator/denominator match between simulator and real runs:
  - dynamic: compare percent error on *normalized end-to-end latency* (`request_e2e_time_normalized`) under a Poisson arrival process (`extern/tracked/vidur/paper/tex/5-eval.tex`)
  - static: exclude scheduling delay and compare *normalized execution-plus-preemption time* (`request_execution_plus_preemption_time_normalized`) (`extern/tracked/vidur/paper/tex/5-eval.tex`)
- They run dynamic workloads at a comparable operating regime by using 85% of maximum serving capacity for each scenario (`extern/tracked/vidur/paper/tex/figures-tex/fig-fidelity-dynamic-trace.tex`).
  - Capacity is defined via a scheduling-delay constraint (P99 scheduling delay under 5s) in the paper’s benchmark/search discussion (`extern/tracked/vidur/paper/tex/4-benchmark.tex`), which matches Vidur’s scheduling-delay metric definition (`extern/tracked/vidur/docs/metrics.md`).
- For overhead control, the paper explicitly states they use “an optimized version of the vLLM codebase” with CUDA graphs and “an extensive telemetry system” (`extern/tracked/vidur/paper/tex/5-eval.tex`) so measurement is **in-engine** (not an external Python client timer).

Out of scope / missing detail:

- The TeX in this repo does not quantify telemetry overhead or spell out the exact hook points used to record `a_r,s_r,f_r,c_r` in the real engine.
- Minimal next step to confirm: locate the corresponding telemetry implementation in the real engine code used by the paper (not vendored here) and document (a) where timestamps are captured and (b) what configs disable expensive tracing.
- Where to document it: `specs/002-reproduce-vidur-paper-fidelity/research.md` (summary) and/or a longer runbook under `context/`.

## According to the current plan, how do we make sure Sarathi-Serve measures the same thing as Vidur?
> Last revised at: `2026-01-05T03:27:58Z` | Last revised base commit: `b735cecbb66fde6f6f956c68ec34f59877e0725a`

- Drive both sim and real with the **same canonical trace** (`trace.csv` with `arrived_at,num_prefill_tokens,num_decode_tokens`) so the request stream is identical (`specs/002-reproduce-vidur-paper-fidelity/plan.md`, `specs/002-reproduce-vidur-paper-fidelity/research.md`).
- For Vidur, preserve Vidur’s **paper-required columns** in the sim artifacts (don’t recompute/rename away the normalized metrics):
  - keep `request_scheduling_delay`, `request_execution_plus_preemption_time_normalized`, and `request_e2e_time_normalized` from Vidur’s raw `request_metrics.csv`
  - implement via extending `src/gpu_simulate_test/vidur_ext/sim_runner.py` as scoped in `specs/002-reproduce-vidur-paper-fidelity/tasks.md`
- For Sarathi-Serve, avoid client-side “TTFT-as-scheduling-delay” approximations; instead use **engine-native request lifecycle metrics** consistent with Vidur’s definitions (see `extern/tracked/sarathi-serve/sarathi/metrics/README.md`):
  - `request_scheduling_delay` corresponds to `s_r - a_r`
  - `request_e2e_time_normalized` corresponds to `(c_r - a_r) / output_tokens`
  - `request_execution_plus_preemption_time_normalized` corresponds to `(c_r - s_r) / output_tokens`
- Keep instrumentation overhead low and comparable by default:
  - prefer in-engine metrics output (Sarathi metrics store) over per-token Python timing hooks
  - disable expensive tracing unless explicitly requested by a scenario (`enable_chrome_trace`, per-token lists, etc.)
- Enforce comparability in scoring by requiring both sim and real metrics files contain the same scenario id + required columns, then compute percent error on the paper’s metrics/percentiles (`specs/002-reproduce-vidur-paper-fidelity/spec.md`, `specs/002-reproduce-vidur-paper-fidelity/contracts/paper_fidelity.openapi.yaml`).
