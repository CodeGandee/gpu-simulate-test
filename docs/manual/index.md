# Manual

This section is for users who want to run the repo’s workflows end-to-end.

## What you get

Two workflows are supported:

### A) Compare Vidur vs real timing (`001-compare-vidur-real-timing`)

- Deterministic workload spec generation (shared by real + simulated runs)
- Real timing runs with either:
  - `backend=transformers` (HuggingFace `generate` with a token streamer), or
  - `backend=sarathi` (Sarathi-Serve engine)
- Vidur simulation driven by the same workload spec and a profiling bundle
- A comparison report (TTFT + decode-token latency percentiles + plots)

### B) Paper fidelity reproduction (`002-reproduce-vidur-paper-fidelity`)

- Canonical trace generation (`trace.csv`) and validation (static or Poisson-arrival dynamic traces)
- Optional trace subsetting for fast iteration (range `[begin,end)` or discrete indices for untimed sources)
- Vidur simulation + Sarathi-Serve real replay over the same trace
- Capacity discovery for dynamic 85% operating point runs
- Scoring and a human-readable report under `results/reports/...`

## Key entrypoints

- Compare workflow (001):
  - Commands (Pixi tasks): `workload-spec`, `real-bench`, `vidur-profile`, `vidur-sim`, `compare-runs` (see `pyproject.toml`)
  - Spec + contracts: `specs/001-compare-vidur-real-timing/`
- Paper fidelity workflow (002):
  - Command (Pixi task): `paper-fidelity` (subcommands: `trace`, `repro`, `profile`, `score`)
  - Spec: `specs/002-reproduce-vidur-paper-fidelity/`
