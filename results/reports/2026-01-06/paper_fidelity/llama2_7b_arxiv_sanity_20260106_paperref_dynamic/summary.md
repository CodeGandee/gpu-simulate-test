# Paper Fidelity Report: llama2_7b_arxiv_sanity_20260106_paperref_dynamic

## Inputs
- sim: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sanity_20260106_paperref_dynamic/sim/request_metrics.csv`
- real: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sanity_20260106_paperref_dynamic/real/request_metrics.csv`

## Profiling
- root: `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur`
- mode: `paper`
- interpretation: sanity-check reproduction (paper-provided profiling bundle)

## Paper Reference
- model: `LLaMA2-7B (TP1)`
- trace: `Arxiv-4K`
- series: `predicted`
- load_frac_of_capacity: `0.85`
- sources: `/data1/huangzhe/code/gpu-simulate-test/context/summaries/vidur-kb/paper-results/dynamic_fidelity_v8_request_e2e_time_normalized_85_p50.json`, `/data1/huangzhe/code/gpu-simulate-test/context/summaries/vidur-kb/paper-results/dynamic_fidelity_v8_request_e2e_time_normalized_85_p95.json`

## Scores
| Metric | Percentile | Paper | Sim | Real | Sim vs Paper | Sim vs Real | Verdict |
|--------|------------|-------|-----|------|--------------|-------------|---------|
| request_execution_plus_preemption_time_normalized | p50 | N/A | 0.0121794 | 0.0312448 | N/A | 61.02% | fail |
| request_execution_plus_preemption_time_normalized | p95 | N/A | 0.0155421 | 0.0536616 | N/A | 71.04% | fail |
| request_e2e_time_normalized | p50 | 0.0359902 | 0.0121805 | 0.0602457 | 66.16% | 79.78% | fail |
| request_e2e_time_normalized | p95 | 0.0412453 | 0.0160417 | 0.111655 | 61.11% | 85.63% | fail |

## Gap Diagnosis
- Vidur sim may underpredict wall-clock latency when CPU/runtime overhead is excluded; see `context/issues/known/issue-vidur-sim-underpredicts-sarathi-real-qwen3-0.6b.md`.
- Sim metrics: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sanity_20260106_paperref_dynamic/sim/request_metrics.csv`; Real metrics: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sanity_20260106_paperref_dynamic/real/request_metrics.csv`.

