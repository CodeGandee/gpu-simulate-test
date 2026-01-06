# Paper Fidelity Report: llama2_7b_arxiv_sanity_20260106_0248

## Inputs
- sim: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sanity_20260106_0248/sim/request_metrics.csv`
- real: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sanity_20260106_0248/real/request_metrics.csv`

## Profiling
- root: `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur`
- mode: `paper`
- interpretation: sanity-check reproduction (paper-provided profiling bundle)

## Scores
| Metric | Percentile | Sim | Real | Percent error | Verdict |
|--------|------------|-----|------|---------------|---------|
| request_execution_plus_preemption_time_normalized | p50 | 0.0103513 | 0.0193468 | 46.50% | fail |
| request_execution_plus_preemption_time_normalized | p95 | 0.0108625 | 0.0226121 | 51.96% | fail |
| request_e2e_time_normalized | p50 | 0.0106497 | 0.0383716 | 72.25% | fail |
| request_e2e_time_normalized | p95 | 0.0108625 | 0.0437724 | 75.18% | fail |

## Gap Diagnosis
- Vidur sim may underpredict wall-clock latency when CPU/runtime overhead is excluded; see `context/issues/known/issue-vidur-sim-underpredicts-sarathi-real-qwen3-0.6b.md`.
- Sim metrics: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sanity_20260106_0248/sim/request_metrics.csv`; Real metrics: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sanity_20260106_0248/real/request_metrics.csv`.

