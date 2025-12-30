# Contracts: Compare Vidur vs real Qwen3 A100 timing

These contracts define the canonical on-disk file formats used by the workflow.

All timestamps are **integer nanoseconds** (`*_ns`) relative to the run/workload start and should be interpreted as monotonic time.

## Files

- Workload spec:
  - `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/contracts/workload_spec.md`
  - `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/contracts/workload_meta.schema.json`
- Run outputs:
  - `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/contracts/run_meta.schema.json`
  - `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/contracts/request_metrics.md`
  - `/data1/huangzhe/code/gpu-simulate-test/specs/001-compare-vidur-real-timing/contracts/token_metrics.md`
