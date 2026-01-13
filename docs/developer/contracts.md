# Data contracts

The canonical contract docs live under:

- Compare workflow: `specs/001-compare-vidur-real-timing/contracts/`
- Paper fidelity workflow: `specs/002-reproduce-vidur-paper-fidelity/contracts/`

Both workflows emit stable artifact families under `tmp/` (heavy outputs) and `results/` (human-readable reports).

## Compare workflow artifacts (`001-compare-vidur-real-timing`)

### Workload spec

Directory: `tmp/workloads/<workload_id>/`

- `trace_lengths.csv`: per-request token counts
- `trace_intervals.csv`: per-request arrivals in nanoseconds

### Run metrics (real or sim)

Directory: `tmp/real_runs/<run_id>/` or `tmp/vidur_runs/<run_id>/`

- `request_metrics.csv`: one row per request (TTFT, completion, token counts)
- `token_metrics.csv`: long-format token times and per-token deltas
- `run_meta.json`: provenance (resolved config, git info, env snapshot)

### Comparison report

Directory: `tmp/comparisons/<comparison_id>/`

- `summary.md`
- `tables/*.csv`
- `figs/*`

## Paper fidelity artifacts (`002-reproduce-vidur-paper-fidelity`)

### Canonical trace

Directory: `tmp/paper_fidelity/traces/<scenario>/`

- `trace.csv`: canonical input schema (`arrived_at,num_prefill_tokens,num_decode_tokens`, optional ids)
- `trace_meta.json`: provenance (workload mode, seed, subset selection; may include trace source info and/or chosen dynamic QPS)

### Runs (sim + real)

Directory: `tmp/paper_fidelity/runs/<scenario>/`

- `sim/request_metrics.csv`: Vidur paper metric columns (normalized metrics preserved from Vidur outputs)
- `real/request_metrics.csv`: Sarathi paper metric columns (converted from `sequence_metrics.csv`)
- `capacity/capacity.json`: capacity discovery outputs (dynamic repro)

### Report

Directory: `results/reports/<date>/paper_fidelity/<report_scenario>/`

- `summary.md`
- `run_meta.json`
- `scores.json`
- `inputs/` (snapshots of sim/real CSVs; trace/capacity inputs when available)
- `figs/` and `tables/`

Naming note: `<report_scenario>` is `scenario.name` for static runs, and `scenario.name_dynamic_<scale>` for dynamic runs.

If you change schemas, update both:

- the contract docs (`specs/.../contracts/`)
- schema validation tests (`tests/unit/`)
