# Advanced tutorial: Vidur `profiling.mlp.profile_method` sweep (sim-vs-real)

This tutorial shows how Vidur’s MLP kernel profiling method (`profiling.mlp.profile_method`) impacts **sim-vs-real**
accuracy for the `vidur-cli` pipeline.

It runs the same **static** trace (this directory’s `inputs/trace_import.csv`) through:

- `profiling.mlp.profile_method=cuda_event`
- `profiling.mlp.profile_method=record_function`
- `profiling.mlp.profile_method=kineto`
- `profiling.mlp.profile_method=perf_counter`

and then summarizes the resulting sim-vs-real score tables into a single comparison.

## Quickstart (run the sweep)

From repo root:

```bash
docs/tutorial/in-depth/adv-tut-vidur-cli-mlp-profile-methods/run_sweep_static_profile_methods.sh
```

This writes all artifacts under a fresh directory in `tmp/` and prints the path.

## Outputs (what to look at)

The sweep directory contains:

- `comparison.md` (human-readable table)
- `comparison_scores.csv` (full score table per method/metric/percentile)
- `comparison_runs.csv` (run directories + method settings)
- `*_summary.md` (copied report summaries, one per method)
- `*.log` (per-method run logs)

## Expected outputs (tracked)

`expected_outputs/` contains a small, sanitized example of:

- `comparison.md`
- `comparison_scores.csv`
- `comparison_runs.csv`

Exact numbers will vary across machines (and across code revisions); the goal is to show the **shape** of the
comparison and the kind of differences you should expect.

## Prerequisites

- Pixi env is set up: `pixi install`
- Submodules are initialized: `git submodule update --init --recursive`
- GPU works in Pixi: `pixi run python -c "import torch; print(torch.cuda.is_available())"`
- LLaMA2-7B model ref exists: `bash models/llama2-7b-hf/bootstrap.sh`

## Maintainers: refresh `expected_outputs/`

Run:

```bash
docs/tutorial/in-depth/adv-tut-vidur-cli-mlp-profile-methods/run_sweep_static_profile_methods.sh --snapshot-expected
```

Then copy the generated sanitized snapshot into:

- `docs/tutorial/in-depth/adv-tut-vidur-cli-mlp-profile-methods/expected_outputs/`

