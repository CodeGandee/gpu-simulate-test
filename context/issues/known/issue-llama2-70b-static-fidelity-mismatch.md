# LLaMA2-70B (TP4) static-fidelity mismatch vs paper (Arxiv-4K)

## Summary

For the Vidur paper’s static fidelity figure (`request_execution_plus_preemption_time_normalized`, P50/P95), the **LLaMA2-70B (TP4) + Arxiv-4K** case is currently **not perfectly reproducible** with our “vendor artifacts → Vidur sim” pipeline:

- Paper (P50, predicted, Arxiv-4K): ~`0.487 s/token` (`context/summaries/vidur-kb/paper-results/static_fidelity_v12_request_execution_plus_preemption_time_normalized_p50.json`).
- Our runs can land **far away** depending on the scheduler/config:
  - **Too low** when `request_preemption_time` is ~0 (e.g., `sarathi` scheduler): ~`0.209 s/token`.
  - **Too high** when decode is heavily “paused” (large `request_preemption_time`, e.g., `vllm` with large `batch_size_cap`): ~`0.98 s/token`.
  - **Best-known (as of Phase 3 report)** with vLLM scheduler + paper-derived knobs: ~`0.312 s/token` (still below paper).

Other models are much less sensitive, so they appear to “match” while 70B(TP4) remains an outlier.

## What we changed (and why it got closer)

Based on `context/tasks/done/002-reproduce-vidur-paper-fidelity/qa-impl-phase-3-repro-report.md`, we tightened the reproduction to better match the paper’s static-fidelity setup and reduced the extreme variability caused by scheduler/config drift:

- **Use the paper’s static metric + setting explicitly**: focus on `request_execution_plus_preemption_time_normalized` under **static arrivals** with the **vLLM scheduler** (the paper states “default vLLM scheduler” for fidelity).
- **Apply “optimal deployment configuration” knobs from the paper**: source `tp_dim` and `batch_size` from `context/summaries/vidur-kb/paper-configs.json`, and map `batch_size` to vLLM’s `batch_size_cap`.
- **Force vLLM scheduler for comparability**: avoid mixing `sarathi`-scheduler runs (where `request_preemption_time` can collapse to ~0 for this metric) with vLLM-scheduler runs.
- **Run controlled sensitivity overrides**: set **PP=1** and **A100** across runs to isolate scheduler/knob effects (note: the paper’s “optimal config” table indicates `Llama-70b + Arxiv` is `H100`, so forcing `A100` is a deliberate deviation for controlled comparison).

### Updated status (narrowed gap, still not fully matched)

- With the “optimal-config” rerun (vLLM scheduler; static arrivals; A100; PP=1), the LLaMA2-70B (TP4) + Arxiv-4K P50 moved from the earlier extremes (~`0.209 s/token` “too low” and ~`0.98 s/token` “too high”) to ~`0.312 s/token` (paper predicted P50 is ~`0.487 s/token`).
- Artifacts/figures for the rerun are tracked in `context/tasks/done/002-reproduce-vidur-paper-fidelity/figures/rerun_optimal_config_static_fidelity_request_execution_plus_preemption_time_normalized_p50.svg` and `context/tasks/done/002-reproduce-vidur-paper-fidelity/figures/rerun_optimal_config_static_fidelity_request_execution_plus_preemption_time_normalized_p95.svg`.

## Why this happens (likely)

1. **The metric includes “preemption time”**
   - `request_execution_plus_preemption_time_normalized` is `(execution_time + preempted_time) / num_decode_tokens` (see `extern/tracked/vidur/vidur/metrics/metrics_store.py`).
   - `preempted_time` is driven by **iteration gaps** (i.e., how long a request waits between decode iterations while other work runs) and is highly scheduler-dependent.

2. **Scheduler/config mismatch changes `request_preemption_time` drastically**
   - The paper’s static workload section uses the **vLLM scheduler** and “offline” arrivals (`extern/tracked/vidur/paper/tex/5-eval.tex`).
   - Our “vendor-results” set under `results/raw/vendor-results/sarathi-serve/dynamic/**` was produced with `sarathi` + Poisson arrivals (not comparable to the paper’s static plot).
   - Even when we run “static + vLLM”, the chosen knobs (notably `batch_size_cap`) can over-starve decodes for 70B(TP4), inflating `request_preemption_time` far above what the paper figure implies.

3. **70B(TP4) is unusually sensitive**
   - With TP4 + large KV-cache pressure, the vLLM-style “prefill-first” behavior can create long gaps between decode iterations (large `preempted_time`), while smaller models don’t hit the same regime.

## Evidence

- Paper number (Arxiv-4K, predicted): `0.487372... s/token`
  - `context/summaries/vidur-kb/paper-results/static_fidelity_v12_request_execution_plus_preemption_time_normalized_p50.json`
- Example “too low” run (preemption ~0):
  - `results/raw/vendor-results/sarathi-serve/dynamic/a100-meta-llama__Llama-2-70b-hf-arxiv_summarization_stats_llama2_tokenizer_filtered_v2/*/request_metrics.csv`
  - `request_preemption_time` P50 is ~`0.0`, so the metric collapses to execution-only (~`0.209 s/token`).
- Example “too high” run (large preemption):
  - `results/raw/vendor-results/vllm/static/a100-meta-llama__Llama-2-70b-hf-arxiv_summarization_stats_llama2_tokenizer_filtered_v2/*/request_metrics.csv`
  - `request_preemption_time` P50 is large (~`128s`), pushing the normalized metric to ~`0.98 s/token`.

## Mitigations / next steps

- Treat the **70B(TP4) static-fidelity** case as a known limitation until we:
  - Identify any **additional fidelity-critical knobs** beyond `tp_dim` and `batch_size` (e.g., other vLLM/Vidur limits that impact decode fairness and therefore `request_preemption_time`).
  - Remove the controlled `A100`/`PP=1` overrides and re-run with the **paper-matched hardware/topology** implied by the “optimal config” row (the paper-configs extract indicates `H100` for `Llama-70b + Arxiv`).
  - Re-run static-fidelity and confirm `request_preemption_time` and the normalized metric land in the paper’s regime.
