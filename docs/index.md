# gpu-simulate-test docs

These docs cover two implemented workflows:

1. `001-compare-vidur-real-timing`: Compare **Vidur CPU-side simulation** outputs against **real GPU inference timing** (per-request + per-token) for a workload spec (initially developed around `Qwen/Qwen3-0.6B` on A100).
2. `002-reproduce-vidur-paper-fidelity`: Reproduce Vidur’s paper-aligned **fidelity metrics** by running the same canonical `trace.csv` through **Vidur simulation** and **Sarathi-Serve real replay**, then scoring percent error and writing a report.

## Sections

- Manual: how to run workflows and interpret outputs/artifacts.
- Developer: architecture, configs, and how implementations map to specs/tasks.

## Quick commands

```bash
pixi install
pixi run mkdocs serve -a 127.0.0.1:8000
```
