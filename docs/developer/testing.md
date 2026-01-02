# Testing

This feature has both unit tests (CPU-only) and manual smoke scripts (some require an A100).

## Unit tests (CPU-only)

```bash
pixi run pytest tests/unit
```

Useful focused tests:

```bash
pixi run pytest tests/unit/test_workload_spec_determinism.py
pixi run pytest tests/unit/test_real_metrics_schema.py
pixi run pytest tests/unit/test_vidur_sim_prereqs.py
pixi run pytest tests/unit/test_compare_alignment.py
```

## Manual smoke scripts

These write outputs under `tmp/`.

```bash
pixi run python tests/manual/test_workload_spec_smoke.py
pixi run python tests/manual/test_compare_runs_smoke.py
```

GPU required:

```bash
pixi run python tests/manual/test_real_bench_smoke.py --backend transformers --workload-dir tmp/workloads/<workload_id>
pixi run python tests/manual/test_vidur_profile_smoke.py
pixi run python tests/manual/test_vidur_sim_smoke.py --workload-dir tmp/workloads/<workload_id> --profiling-root tmp/vidur_profiling/a100/qwen3_0_6b
```

