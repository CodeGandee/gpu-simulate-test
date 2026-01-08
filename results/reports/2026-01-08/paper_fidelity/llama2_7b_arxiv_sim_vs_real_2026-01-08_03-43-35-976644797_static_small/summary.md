# Paper Fidelity Report: llama2_7b_arxiv_sim_vs_real_2026-01-08_03-43-35-976644797_static_small

## Inputs
- sim: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sim_vs_real_2026-01-08_03-43-35-976644797_static_small/sim/request_metrics.csv`
- real: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sim_vs_real_2026-01-08_03-43-35-976644797_static_small/real/request_metrics.csv`

## Profiling
- root: `/data1/huangzhe/code/gpu-simulate-test/results/raw/vidur-profiling/llama2-7b/sarathi-serve/2026-01-07_15-15-15-281697026`
- mode: `host`
- interpretation: gap reproduction (host profiling bundle under results/raw/vidur-profiling)

## Scores
| Metric | Percentile | Sim | Real | Percent error | Verdict |
|--------|------------|-----|------|---------------|---------|
| request_execution_plus_preemption_time_normalized | p50 | 0.0400585 | 0.0355395 | 12.72% | fail |
| request_execution_plus_preemption_time_normalized | p95 | 0.0467845 | 0.0701834 | 33.34% | fail |
| request_e2e_time_normalized | p50 | 0.0645173 | 0.447524 | 85.58% | fail |
| request_e2e_time_normalized | p95 | 0.116475 | 1.14365 | 89.82% | fail |
| prefill_time_execution_plus_preemption_normalized | p50 | 9.98948e-05 | 0.00111411 | 91.03% | fail |
| prefill_time_execution_plus_preemption_normalized | p95 | 0.000129656 | 0.0012701 | 89.79% | fail |
| decode_time_execution_plus_preemption_normalized | p50 | 0.0382738 | 0.016902 | 126.45% | fail |
| decode_time_execution_plus_preemption_normalized | p95 | 0.0441245 | 0.0180198 | 144.87% | fail |

## Figures
### Static normalized latency
![ECDF: request_execution_plus_preemption_time_normalized](figs/request_execution_plus_preemption_time_normalized_ecdf.svg)
![Percentiles: request_execution_plus_preemption_time_normalized](figs/request_execution_plus_preemption_time_normalized_percentiles.svg)

### Dynamic normalized latency
![ECDF: request_e2e_time_normalized](figs/request_e2e_time_normalized_ecdf.svg)
![Percentiles: request_e2e_time_normalized](figs/request_e2e_time_normalized_percentiles.svg)

## Gap Diagnosis
- Vidur sim may underpredict wall-clock latency when CPU/runtime overhead is excluded and/or the profiling bundle is not host-matched; see `context/issues/known/issue-vidur-sim-underpredicts-sarathi-real.md`.
- Sim metrics: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sim_vs_real_2026-01-08_03-43-35-976644797_static_small/sim/request_metrics.csv`; Real metrics: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sim_vs_real_2026-01-08_03-43-35-976644797_static_small/real/request_metrics.csv`.

