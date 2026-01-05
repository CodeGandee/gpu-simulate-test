# Paper Fidelity Report: adhoc_79e74acf1cdb

## Inputs
- sim: `/data1/huangzhe/code/gpu-simulate-test/tests/fixtures/paper_fidelity/sim_request_metrics.csv`
- real: `/data1/huangzhe/code/gpu-simulate-test/tests/fixtures/paper_fidelity/real_request_metrics.csv`

## Scores
| Metric | Percentile | Sim | Real | Percent error | Verdict |
|--------|------------|-----|------|---------------|---------|
| request_execution_plus_preemption_time_normalized | p50 | 110 | 100 | 10.00% | fail |
| request_execution_plus_preemption_time_normalized | p95 | 110 | 100 | 10.00% | fail |
| request_e2e_time_normalized | p50 | 105 | 100 | 5.00% | pass |
| request_e2e_time_normalized | p95 | 105 | 100 | 5.00% | pass |

## Gap Diagnosis
- Vidur sim may underpredict wall-clock latency when CPU/runtime overhead is excluded; see `context/issues/known/issue-vidur-sim-underpredicts-sarathi-real-qwen3-0.6b.md`.
- Sim metrics: `/data1/huangzhe/code/gpu-simulate-test/tests/fixtures/paper_fidelity/sim_request_metrics.csv`; Real metrics: `/data1/huangzhe/code/gpu-simulate-test/tests/fixtures/paper_fidelity/real_request_metrics.csv`.

