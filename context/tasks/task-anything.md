first read about this:
`context/issues/known/issue-llama2-70b-static-fidelity-mismatch.md`

then double-check the Vidur paper’s claim about the scheduler used for the **fidelity** figures:
- `extern/tracked/vidur/paper/tex/5-eval.tex`
  - in the “Simulator Fidelity” section: *“We use the default vLLM scheduler for all these experiments.”*

and then, read about the optimal configs for each model+dataset in:
- `context/summaries/vidur-kb/paper-configs.md`

then, read about the experiment results in:
- `context/tasks/working/002-reproduce-vidur-paper-fidelity/qa-impl-phase-3-repro-report.md`
- in the question `Which paper metric(s) can we currently reproduce...`

here is what we suspect:
- we are not matching the exact **scheduler knobs** used in the Vidur paper for the fidelity runs (especially the ones that affect `request_preemption_time`), which can materially change the static-fidelity metric.
- for reproduction, **always use the vLLM scheduler** for all model+dataset runs (per the paper statement above), but still follow `context/summaries/vidur-kb/paper-configs.json` for the other configuration knobs:
  - `sku` → Vidur device (`a100` / `h100`)
  - `tp_dim` / `pp_dim`
  - `batch_size` → vLLM `batch_size_cap`
- rerun the simulations under those configs (vLLM everywhere + paper-config knobs) to see if we can reproduce the paper results (P50/P95).

do that, temporary scripts and outputs save to tmp/<subdir>, do not reuse existing subdirs, create new one.

after you rerun the simulations, produce those figures in `qa-impl-phase-3-repro-report.md`, insert the new results there.
