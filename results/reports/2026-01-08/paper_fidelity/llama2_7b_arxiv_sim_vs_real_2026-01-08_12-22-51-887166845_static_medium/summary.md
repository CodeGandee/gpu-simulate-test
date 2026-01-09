# Paper Fidelity Report: llama2_7b_arxiv_sim_vs_real_2026-01-08_12-22-51-887166845_static_medium

## Inputs
- sim: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sim_vs_real_2026-01-08_12-22-51-887166845_static_medium/sim/request_metrics.csv`
- real: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sim_vs_real_2026-01-08_12-22-51-887166845_static_medium/real/request_metrics.csv`

## Profiling
- root: `/data1/huangzhe/code/gpu-simulate-test/results/raw/vidur-profiling/llama2-7b/sarathi-serve/2026-01-07_15-15-15-281697026`
- mode: `host`
- interpretation: gap reproduction (host profiling bundle under results/raw/vidur-profiling)

## Scores
| Metric | Percentile | Sim | Real | Percent error | Verdict |
|--------|------------|-----|------|---------------|---------|
| request_execution_plus_preemption_time_normalized | p50 | 0.0271987 | 0.0360616 | 24.58% | fail |
| request_execution_plus_preemption_time_normalized | p95 | 0.0551296 | 0.0736869 | 25.18% | fail |
| request_e2e_time_normalized | p50 | 3.00257 | 3.94489 | 23.89% | fail |
| request_e2e_time_normalized | p95 | 11.9561 | 15.7215 | 23.95% | fail |
| prefill_time_execution_plus_preemption_normalized | p50 | 0.000907614 | 0.00120311 | 24.56% | fail |
| prefill_time_execution_plus_preemption_normalized | p95 | 0.00108409 | 0.00145347 | 25.41% | fail |
| decode_time_execution_plus_preemption_normalized | p50 | 0.0129235 | 0.0172321 | 25.00% | fail |
| decode_time_execution_plus_preemption_normalized | p95 | 0.0140866 | 0.01878 | 24.99% | fail |

## Figures
### Static normalized latency
![ECDF: request_execution_plus_preemption_time_normalized](figs/request_execution_plus_preemption_time_normalized_ecdf.svg)
![Percentiles: request_execution_plus_preemption_time_normalized](figs/request_execution_plus_preemption_time_normalized_percentiles.svg)

### Dynamic normalized latency
![ECDF: request_e2e_time_normalized](figs/request_e2e_time_normalized_ecdf.svg)
![Percentiles: request_e2e_time_normalized](figs/request_e2e_time_normalized_percentiles.svg)

## Gap Diagnosis
- Vidur sim may underpredict wall-clock latency when CPU/runtime overhead is excluded and/or the profiling bundle is not host-matched; see `context/issues/known/issue-vidur-sim-underpredicts-sarathi-real.md`.
- Sim metrics: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sim_vs_real_2026-01-08_12-22-51-887166845_static_medium/sim/request_metrics.csv`; Real metrics: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sim_vs_real_2026-01-08_12-22-51-887166845_static_medium/real/request_metrics.csv`.

