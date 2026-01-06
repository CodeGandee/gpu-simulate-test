first read about this:
`context/issues/known/issue-llama2-70b-static-fidelity-mismatch.md`

and then, read about the optimal configs for each model+dataset in:
- `context/summaries/vidur-kb/paper-configs.md`

then, read about the experiment results in:
- `context/tasks/working/002-reproduce-vidur-paper-fidelity/qa-impl-phase-3-repro-report.md`
- in the question `Which paper metric(s) can we currently reproduce...`

here is what we suspect:
- we are not matching the exact scheduler knobs used in the Vidur paper for the static-fidelity runs, we shall refer to those configs as the "optimal deployment configurations" and use those configs to derive the static and dynamic latency results (p50/p95)
- that means, we need to rerun the simualtions under those optimal configs to see if we can reproduce the paper results

do that, temporary scripts and outputs save to tmp/<subdir>, do not reuse existing subdirs, create new one.

after you rerun the simulations, produce those figures in `qa-impl-phase-3-repro-report.md`, insert the new results there.