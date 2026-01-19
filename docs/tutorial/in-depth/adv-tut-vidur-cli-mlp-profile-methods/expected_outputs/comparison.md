# Profile method sweep (vidur_cli_mlp_profile_method_sweep_20260119T113945Z_563447)

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
| cuda_event | 42.35% | 37.62% | 38.31% | 40.20% |
| record_function | 14.25% | 13.98% | 12.96% | 14.41% |
| record_function_org | 16.41% | 14.89% | 15.21% | 16.10% |
| kineto | 21.03% | 18.55% | 18.48% | 19.75% |
| perf_counter | 47.66% | 46.71% | 45.08% | 47.27% |

## Full score table
See `comparison_scores.csv`

## Notes
- Each run is a full `init-run → trace → profile → sim → real → report` pipeline.
- `kineto` is expected to be slow due to profiler overhead.
