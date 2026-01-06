# LLaMA2-70B (TP4) static-fidelity mismatch vs paper (Arxiv-4K)

## Summary

For the Vidur paper’s static fidelity figure (`request_execution_plus_preemption_time_normalized`, P50/P95), the **LLaMA2-70B (TP4) + Arxiv-4K** case is currently **not reproducible** with our “vendor artifacts → Vidur sim” pipeline:

- Paper (P50, predicted, Arxiv-4K): ~`0.487 s/token` (`context/summaries/vidur-kb/paper-results/static_fidelity_v12_request_execution_plus_preemption_time_normalized_p50.json`).
- Our runs can land **far away** depending on the scheduler/config:
  - **Too low** when `request_preemption_time` is ~0 (e.g., `sarathi` scheduler): ~`0.209 s/token`.
  - **Too high** when decode is heavily “paused” (large `request_preemption_time`, e.g., `vllm` with large `batch_size_cap`): ~`0.98 s/token`.

Other models are much less sensitive, so they appear to “match” while 70B(TP4) remains an outlier.

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
  - Identify the **exact scheduler knobs** used for the paper’s static-fidelity runs (especially `batch_size_cap` and any related limits).
  - Re-run the static-fidelity vendor simulation with those knobs and confirm `request_preemption_time` is in the expected regime.
