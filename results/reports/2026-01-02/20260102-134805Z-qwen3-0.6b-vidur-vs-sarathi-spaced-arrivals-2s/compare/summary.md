# Compare runs summary

- real: `/data1/huangzhe/code/gpu-simulate-test/tmp/compare_experiments/20260102-134805Z-qwen3-0.6b-vidur-vs-sarathi-spaced-arrivals-2s/real_run`
- sim: `/data1/huangzhe/code/gpu-simulate-test/tmp/compare_experiments/20260102-134805Z-qwen3-0.6b-vidur-vs-sarathi-spaced-arrivals-2s/vidur_run`

## TTFT percentiles (ns)

| metric | real_ns | sim_ns |
| --- | --- | --- |
| p50 | 17283962.5 | 1967610.0 |
| p90 | 20037415.3 | 1967610.0 |
| p99 | 22755979.33 | 1967610.0 |

## Decode token latency percentiles (ns)

| metric | real_ns | sim_ns |
| --- | --- | --- |
| p50 | 13611908.0 | 1824346.0 |
| p90 | 14125543.0 | 1828523.0 |
| p99 | 17128720.2 | 1828524.0 |

Notes:
- Token alignment truncates sim tokens using real `num_decode_tokens_actual` per request.
