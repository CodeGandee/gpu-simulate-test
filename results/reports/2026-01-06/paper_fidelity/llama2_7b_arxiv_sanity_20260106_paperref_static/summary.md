# Paper Fidelity Report: llama2_7b_arxiv_sanity_20260106_paperref_static

## Inputs
- sim: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sanity_20260106_paperref_static/sim/request_metrics.csv`
- real: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sanity_20260106_paperref_static/real/request_metrics.csv`

## Profiling
- root: `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur`
- mode: `paper`
- interpretation: sanity-check reproduction (paper-provided profiling bundle)

## Paper Reference
- model: `LLaMA2-7B (TP1)`
- trace: `Arxiv-4K`
- series: `predicted`
- sources: `/data1/huangzhe/code/gpu-simulate-test/context/summaries/vidur-kb/paper-results/static_fidelity_v12_request_execution_plus_preemption_time_normalized_p50.json`, `/data1/huangzhe/code/gpu-simulate-test/context/summaries/vidur-kb/paper-results/static_fidelity_v12_request_execution_plus_preemption_time_normalized_p95.json`

## Scores
| Metric | Percentile | Paper | Sim | Real | Sim vs Paper | Sim vs Real | Verdict |
|--------|------------|-------|-----|------|--------------|-------------|---------|
| request_execution_plus_preemption_time_normalized | p50 | 0.0747704 | 0.0196407 | 0.0312699 | 73.73% | 37.19% | fail |
| request_execution_plus_preemption_time_normalized | p95 | 0.0812659 | 0.0239597 | 0.0539942 | 70.52% | 55.63% | fail |
| request_e2e_time_normalized | p50 | N/A | 0.0261222 | 0.115861 | N/A | 77.45% | fail |
| request_e2e_time_normalized | p95 | N/A | 0.0341197 | 0.212718 | N/A | 83.96% | fail |

## Gap Diagnosis
- Vidur sim may underpredict wall-clock latency when CPU/runtime overhead is excluded and/or the profiling bundle is not host-matched; see `context/issues/known/issue-vidur-sim-underpredicts-sarathi-real.md`.
- Sim metrics: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sanity_20260106_paperref_static/sim/request_metrics.csv`; Real metrics: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sanity_20260106_paperref_static/real/request_metrics.csv`.
