# Paper Fidelity Report: llama2_7b_arxiv_sanity_20260106_0253_dynamic

## Inputs
- sim: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sanity_20260106_0253_dynamic/sim/request_metrics.csv`
- real: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sanity_20260106_0253_dynamic/real/request_metrics.csv`

## Profiling
- root: `/data1/huangzhe/code/gpu-simulate-test/extern/tracked/vidur`
- mode: `paper`
- interpretation: sanity-check reproduction (paper-provided profiling bundle)

## Scores
| Metric | Percentile | Sim | Real | Percent error | Verdict |
|--------|------------|-----|------|---------------|---------|
| request_execution_plus_preemption_time_normalized | p50 | 0.0100186 | 0.019487 | 48.59% | fail |
| request_execution_plus_preemption_time_normalized | p95 | 0.0104976 | 0.0228951 | 54.15% | fail |
| request_e2e_time_normalized | p50 | 0.0100519 | 0.0291596 | 65.53% | fail |
| request_e2e_time_normalized | p95 | 0.0105815 | 0.0326957 | 67.64% | fail |

## Gap Diagnosis
- Vidur sim may underpredict wall-clock latency when CPU/runtime overhead is excluded; see `context/issues/known/issue-vidur-sim-underpredicts-sarathi-real-qwen3-0.6b.md`.
- Sim metrics: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sanity_20260106_0253_dynamic/sim/request_metrics.csv`; Real metrics: `/data1/huangzhe/code/gpu-simulate-test/tmp/paper_fidelity/runs/llama2_7b_arxiv_sanity_20260106_0253_dynamic/real/request_metrics.csv`.

