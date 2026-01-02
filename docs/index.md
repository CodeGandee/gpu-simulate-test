# gpu-simulate-test docs

These docs focus on the `001-compare-vidur-real-timing` feature: a Pixi + Hydra workflow to compare **Vidur CPU-side simulation** outputs against **real GPU inference timing** for `Qwen/Qwen3-0.6B` on A100.

## Sections

- Manual: how to run the workflow and interpret outputs.
- Developer: architecture, configs, and how the implementation maps to `specs/001-compare-vidur-real-timing/tasks.md`.

## Quick commands

```bash
pixi install
pixi run mkdocs serve -a 127.0.0.1:8000
```

