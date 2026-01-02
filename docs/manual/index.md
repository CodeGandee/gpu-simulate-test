# Manual

This section is for users who want to run the `001-compare-vidur-real-timing` workflow end-to-end.

## What you get

- Deterministic workload spec generation (shared by real + simulated runs)
- Real timing runs with either:
  - `backend=transformers` (HuggingFace `generate` with a token streamer), or
  - `backend=sarathi` (Sarathi-Serve engine)
- Vidur simulation driven by the same workload spec and a profiling bundle
- A comparison report (TTFT + decode-token latency percentiles + plots)

## Key entrypoints

- Commands (Pixi tasks): `workload-spec`, `real-bench`, `vidur-profile`, `vidur-sim`, `compare-runs` (see `pyproject.toml`)
- Feature spec and contracts: `specs/001-compare-vidur-real-timing/`

