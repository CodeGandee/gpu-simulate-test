# Data contracts

The canonical contract docs live under:

- `specs/001-compare-vidur-real-timing/contracts/`

The workflow emits three main artifact families:

## Workload spec

Directory: `tmp/workloads/<workload_id>/`

- `trace_lengths.csv`: per-request token counts
- `trace_intervals.csv`: per-request arrivals in nanoseconds

## Run metrics (real or sim)

Directory: `tmp/real_runs/<run_id>/` or `tmp/vidur_runs/<run_id>/`

- `request_metrics.csv`: one row per request (TTFT, completion, token counts)
- `token_metrics.csv`: long-format token times and per-token deltas
- `run_meta.json`: provenance (resolved config, git info, env snapshot)

## Comparison report

Directory: `tmp/comparisons/<comparison_id>/`

- `summary.md`
- `tables/*.csv`
- `figs/*`

If you change schemas, update both:

- the contract docs (`specs/.../contracts/`)
- schema validation tests (`tests/unit/`)

