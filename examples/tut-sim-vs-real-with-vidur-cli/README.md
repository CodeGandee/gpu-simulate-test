# Demo: `vidur-cli` sim-vs-real (static) from a paper-fidelity trace snapshot

This folder is a **self-contained, git-tracked** demo for the tutorial:

- `docs/tutorial/howto/tut-sim-vs-real-with-vidur-cli.md`

It demonstrates an end-to-end `vidur-cli` run for **LLaMA2-7B** on **A100** using the **Sarathi** real backend, producing a final sim-vs-real report.

## What’s tracked here (inputs + expected outputs)

Inputs:

- `inputs/trace.csv`: paper-fidelity-style trace snapshot (`arrived_at` seconds). This was copied from:
  - `results/reports/2026-01-15/paper_fidelity/llama2_7b_arxiv_vc_profile_static_small_20260115/inputs/trace.csv`
- `inputs/trace_import.csv`: `vidur-cli` canonical trace import format (`arrival_time_ns` nanoseconds).

Expected output snapshot (example):

- `expected_report/`: a representative `<run_dir>/report/` directory snapshot from one successful run on this repo.
  - Exact **numbers** may vary across machines.
  - Machine-local paths are sanitized to placeholders like `<RUN_DIR>` and `<MODEL_REF>`.
  - The key goal is that the **artifact structure** matches and the report includes the **apple-to-apple** config section.

## How to run the demo

From anywhere:

```bash
examples/tut-sim-vs-real-with-vidur-cli/run_demo_static_from_pf_trace.sh
```

The script:

- pins GPUs via `GSIM_CUDA_VISIBLE_DEVICES` (defaults to `4,5`)
- creates a fresh workspace under `<repo>/tmp/`
- runs: `init-run → trace(import) → profile → sim → real → report`
- prints the final report path (`<run_dir>/report/summary.md`)

## Maintainers: refresh `expected_report/`

```bash
examples/tut-sim-vs-real-with-vidur-cli/run_demo_static_from_pf_trace.sh --refresh-expected-report
```

This overwrites `expected_report/` with the newly produced `<run_dir>/report/` directory.
Machine-local paths are sanitized after copying (so the snapshot is stable to commit).
