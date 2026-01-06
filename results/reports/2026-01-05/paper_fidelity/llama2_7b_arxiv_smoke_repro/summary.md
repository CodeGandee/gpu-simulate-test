# Paper Fidelity Report: llama2_7b_arxiv_smoke_repro

## Inputs
- sim: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_smoke_repro/sim/request_metrics.csv`
- real: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_smoke_repro/real/request_metrics.csv`

## Scores
| Metric | Percentile | Sim | Real | Percent error | Verdict |
|--------|------------|-----|------|---------------|---------|
| request_execution_plus_preemption_time_normalized | p50 | 0.00935125 | 0.0309931 | 69.83% | fail |
| request_execution_plus_preemption_time_normalized | p95 | 0.00935125 | 0.0309931 | 69.83% | fail |
| request_e2e_time_normalized | p50 | 0.00935125 | 0.0310743 | 69.91% | fail |
| request_e2e_time_normalized | p95 | 0.00935125 | 0.0310743 | 69.91% | fail |

## Gap Diagnosis
- Vidur sim may underpredict wall-clock latency when CPU/runtime overhead is excluded and/or the profiling bundle is not host-matched; see `context/issues/known/issue-vidur-sim-underpredicts-sarathi-real.md`.
- Sim metrics: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_smoke_repro/sim/request_metrics.csv`; Real metrics: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_smoke_repro/real/request_metrics.csv`.
