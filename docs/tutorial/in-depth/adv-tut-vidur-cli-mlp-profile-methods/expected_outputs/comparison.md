# Profile method sweep (example)

## Runs
| method | run_dir |
|--------|---------|
| cuda_event | `<RUN_DIR_cuda_event>` |
| record_function | `<RUN_DIR_record_function>` |
| record_function_org | `<RUN_DIR_record_function_org>` |
| kineto | `<RUN_DIR_kineto>` |
| perf_counter | `<RUN_DIR_perf_counter>` |

## Percent error (selected metrics)
| method | exec+pree p50 | exec+pree p95 | e2e p50 | e2e p95 |
|--------|--------------:|--------------:|--------:|--------:|
| cuda_event | 36.36% | 39.38% | 39.28% | 39.05% |
| record_function | 10.13% | 15.15% | 12.50% | 12.49% |
| record_function_org | N/A | N/A | N/A | N/A |
| kineto | 12.73% | 18.93% | 14.64% | 14.86% |
| perf_counter | 45.12% | 55.96% | 48.45% | 47.16% |

## Full score table
See `comparison_scores.csv`

## Notes
- Each run is a full `init-run → trace → profile → sim → real → report` pipeline.
- `kineto` is expected to be slow due to profiler overhead.
