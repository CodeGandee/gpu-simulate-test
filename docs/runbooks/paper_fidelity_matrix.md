# Runbook: paper-fidelity matrix outputs (deprecated)

The `paper-fidelity matrix` subcommand has been removed.

Use the sweep script instead:

```bash
bash scripts/paper_fidelity_sweep.sh --scale small --workloads static,dynamic --tp 1 --pp 1 --run-id my_run_001
```

See the replacement runbook:

- `docs/runbooks/paper_fidelity_sweep.md`
