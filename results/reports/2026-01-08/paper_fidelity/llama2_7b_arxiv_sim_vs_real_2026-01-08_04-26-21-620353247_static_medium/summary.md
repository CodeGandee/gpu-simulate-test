# Paper Fidelity Report: llama2_7b_arxiv_sim_vs_real_2026-01-08_04-26-21-620353247_static_medium

## Inputs
- sim: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sim_vs_real_2026-01-08_04-26-21-620353247_static_medium/sim/request_metrics.csv`
- real: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sim_vs_real_2026-01-08_04-26-21-620353247_static_medium/real/request_metrics.csv`

## Profiling
- root: `/data1/huangzhe/code/gpu-simulate-test/results/raw/vidur-profiling/llama2-7b/sarathi-serve/2026-01-07_15-15-15-281697026`
- mode: `host`
- interpretation: gap reproduction (host profiling bundle under results/raw/vidur-profiling)

## Scores
| Metric | Percentile | Sim | Real | Percent error | Verdict |
|--------|------------|-----|------|---------------|---------|
| request_execution_plus_preemption_time_normalized | p50 | 0.0553512 | 0.0358409 | 54.44% | fail |
| request_execution_plus_preemption_time_normalized | p95 | 0.0598889 | 0.0731682 | 18.15% | fail |
| request_e2e_time_normalized | p50 | 0.401243 | 3.92011 | 89.76% | fail |
| request_e2e_time_normalized | p95 | 1.53367 | 15.6166 | 90.18% | fail |
| prefill_time_execution_plus_preemption_normalized | p50 | 0.000133803 | 0.00119103 | 88.77% | fail |
| prefill_time_execution_plus_preemption_normalized | p95 | 0.000186472 | 0.0014562 | 87.19% | fail |
| decode_time_execution_plus_preemption_normalized | p50 | 0.0532178 | 0.0170813 | 211.56% | fail |
| decode_time_execution_plus_preemption_normalized | p95 | 0.0550443 | 0.0187855 | 193.01% | fail |

## Figures
### Static normalized latency
![ECDF: request_execution_plus_preemption_time_normalized](figs/request_execution_plus_preemption_time_normalized_ecdf.svg)
![Percentiles: request_execution_plus_preemption_time_normalized](figs/request_execution_plus_preemption_time_normalized_percentiles.svg)

### Dynamic normalized latency
![ECDF: request_e2e_time_normalized](figs/request_e2e_time_normalized_ecdf.svg)
![Percentiles: request_e2e_time_normalized](figs/request_e2e_time_normalized_percentiles.svg)

## Gap Diagnosis
- Vidur sim may underpredict wall-clock latency when CPU/runtime overhead is excluded and/or the profiling bundle is not host-matched; see `context/issues/known/issue-vidur-sim-underpredicts-sarathi-real.md`.
- Sim metrics: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sim_vs_real_2026-01-08_04-26-21-620353247_static_medium/sim/request_metrics.csv`; Real metrics: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sim_vs_real_2026-01-08_04-26-21-620353247_static_medium/real/request_metrics.csv`.

