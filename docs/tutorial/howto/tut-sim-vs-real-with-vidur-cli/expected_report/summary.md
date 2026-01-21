# Sim-vs-Real Report: <RUN_TAG>

## Inputs
- sim: `<RUN_DIR>/report/inputs/sim_request_metrics.csv`
- real: `<RUN_DIR>/report/inputs/real_request_metrics.csv`

## Profiling
- root: `<RUN_DIR>/profile`
- settings:
  - num_gpus: `1`
  - tensor_parallel_size: `1`
  - max_tokens: `4096`
  - include_network: `True`
- cpu_overhead:
  - modeling: `enabled`
  - network_device: `a100_pairwise_nvlink`
  - max_batch_size: `128`
  - validation: `strict`
  - csv: `<RUN_DIR>/profile/data/profiling/cpu_overhead/a100_pairwise_nvlink/meta-llama/Llama-2-7b-hf/cpu_overheads.csv`
  - status: `ok`
- attention: `profile_mode=both backend=FLASHINFER block_size=16 min_batch_size=1 max_batch_size=1`
- mlp:
  - profile_method: `record_function`
  - validation: `mode=strict nan_policy=zero small_input_threshold=128 zero_heavy_limit=0.01`
  - fallback: `enabled=false method=cuda_event used=false`
- mlp_consumer:
  - nan_policy: `zero` (nan_policy=`zero` mode=`strict`)
  - nan_drop: `disabled`
  - nan_fill_zero: `enabled` models_with_fills=`0` cells_filled_total=`0`

## Config (apple-to-apple)
- model_id: `meta-llama/Llama-2-7b-hf`
- model_ref (sim): `<MODEL_REF>`
- model_ref (real): `<MODEL_REF>`
- arrival_kind: `fixed_interval`
- arrival_params: `seed=0`
- trace: num_requests=`50` max_total_tokens=`4021`
- max_tokens: sim=`4096` real=`4096` (match)
- ignore_eos (real): `True`
- tensor_parallel_size: sim=`1` real=`1` (match)
- pipeline_parallel_size: sim=`1` real=`1` (match)
- chunk_size: sim=`16` real=`16` (match)
- batch_size: sim=`16` real=`16` (match)
- block_size (sim): `16`
- watermark_blocks_fraction (sim): `0.01`
- cpu_overhead_modeling (sim): `enabled`

## Scores
| Metric | Percentile | Sim | Real | Percent error |
|--------|------------|-----|------|---------------|
| request_execution_plus_preemption_time_normalized | p50 | 0.0412782 | 0.0346094 | 19.27% |
| request_execution_plus_preemption_time_normalized | p95 | 0.0875419 | 0.0693542 | 26.22% |
| request_e2e_time_normalized | p50 | 0.553043 | 0.454161 | 21.77% |
| request_e2e_time_normalized | p95 | 1.38988 | 1.14332 | 21.57% |
| prefill_time_execution_plus_preemption_normalized | p50 | 0.00135058 | 0.00110354 | 22.39% |
| prefill_time_execution_plus_preemption_normalized | p95 | 0.0014346 | 0.00123789 | 15.89% |
| decode_time_execution_plus_preemption_normalized | p50 | 0.0197568 | 0.0168149 | 17.50% |
| decode_time_execution_plus_preemption_normalized | p95 | 0.0202029 | 0.017412 | 16.03% |

## Figures
### Metric: request_execution_plus_preemption_time_normalized
![ECDF: request_execution_plus_preemption_time_normalized](figs/request_execution_plus_preemption_time_normalized_ecdf.svg)
![Percentiles: request_execution_plus_preemption_time_normalized](figs/request_execution_plus_preemption_time_normalized_percentiles.svg)

### Metric: request_e2e_time_normalized
![ECDF: request_e2e_time_normalized](figs/request_e2e_time_normalized_ecdf.svg)
![Percentiles: request_e2e_time_normalized](figs/request_e2e_time_normalized_percentiles.svg)

### Metric: prefill_time_execution_plus_preemption_normalized
![ECDF: prefill_time_execution_plus_preemption_normalized](figs/prefill_time_execution_plus_preemption_normalized_ecdf.svg)
![Percentiles: prefill_time_execution_plus_preemption_normalized](figs/prefill_time_execution_plus_preemption_normalized_percentiles.svg)

### Metric: decode_time_execution_plus_preemption_normalized
![ECDF: decode_time_execution_plus_preemption_normalized](figs/decode_time_execution_plus_preemption_normalized_ecdf.svg)
![Percentiles: decode_time_execution_plus_preemption_normalized](figs/decode_time_execution_plus_preemption_normalized_percentiles.svg)
