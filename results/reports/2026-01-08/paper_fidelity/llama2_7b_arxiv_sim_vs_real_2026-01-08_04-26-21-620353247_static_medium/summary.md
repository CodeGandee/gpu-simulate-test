# Paper Fidelity Report: llama2_7b_arxiv_sim_vs_real_2026-01-08_04-26-21-620353247_static_medium

## Inputs
- sim: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sim_vs_real_2026-01-08_04-26-21-620353247_static_medium/sim/request_metrics.csv`
- real: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sim_vs_real_2026-01-08_04-26-21-620353247_static_medium/real/request_metrics.csv`

## Profiling
- root: `/data1/huangzhe/code/gpu-simulate-test/results/raw/vidur-profiling/llama2-7b/sarathi-serve/2026-01-07_15-15-15-281697026`
- mode: `host`
- interpretation: gap reproduction (host profiling bundle under results/raw/vidur-profiling)

## Paper Reference
- model: `LLaMA2-7B (TP1)`
- trace: `Arxiv-4K`
- series: `predicted`
- sources: `/data1/huangzhe/code/gpu-simulate-test/context/summaries/vidur-kb/paper-results/static_fidelity_v12_request_execution_plus_preemption_time_normalized_p50.json`, `/data1/huangzhe/code/gpu-simulate-test/context/summaries/vidur-kb/paper-results/static_fidelity_v12_request_execution_plus_preemption_time_normalized_p95.json`

## Scores
| Metric | Percentile | Paper | Sim | Real | Sim vs Paper | Sim vs Real | Verdict |
|--------|------------|-------|-----|------|--------------|-------------|---------|
| request_execution_plus_preemption_time_normalized | p50 | 0.0747704 | 0.0553512 | 0.0358409 | 25.97% | 54.44% | fail |
| request_execution_plus_preemption_time_normalized | p95 | 0.0812659 | 0.0598889 | 0.0731682 | 26.31% | 18.15% | fail |
| request_e2e_time_normalized | p50 | N/A | 0.401243 | 3.92011 | N/A | 89.76% | fail |
| request_e2e_time_normalized | p95 | N/A | 1.53367 | 15.6166 | N/A | 90.18% | fail |

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

