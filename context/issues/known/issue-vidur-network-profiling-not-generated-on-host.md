# Issue: host network profiling isn’t implemented in our Vidur profiling flow

## Summary

In this repo, the `vidur-profile` / `vidur-cli svr profile` stage does not *run* Vidur’s network (collectives)
profilers. Instead, it only copies prebuilt `all_reduce.csv` / `send_recv.csv` from the Vidur submodule’s
`extern/tracked/vidur/data/profiling/network/<network_device>/` directory when available.

As a result, “network profiling on the current host” is not actually supported end-to-end in the current
implementation.

## Impact

- For **TP=1 and PP=1**: usually no impact (Vidur won’t load network collectives models).
- For **TP>1 and/or PP>1**:
  - If you set `hardware.network_device` to a value that does not exist under Vidur’s vendored `data/profiling/network`,
    your profiling root will be missing required network CSVs and simulations can fail when the predictor tries to load
    them.
  - If you keep the default `hardware.network_device` (e.g. `a100_pairwise_nvlink`) but your actual topology is
    different (e.g. a DGX-style full-NVLink fabric), network time modeling can be significantly wrong.

## Repro (one example)

1) Run a profiling stage with a custom `network_device`:

```bash
pixi run vidur-profile \
  profiling.include_network=true \
  hardware.network_device=myhost_custom \
  ...
```

2) Observe the profiling root does not contain:

```text
data/profiling/network/myhost_custom/all_reduce.csv
data/profiling/network/myhost_custom/send_recv.csv
```

3) Run a simulation that requires collectives modeling (TP>1 or PP>1) using that profiling root and network_device.
Vidur can then fail while loading/training the network models, or silently use a mismatched network_device if you
changed it to “make things run”.

## Root cause

`src/gpu_simulate_test/vidur_ext/profile_runner.py` currently stages network CSVs by copying from:

`extern/tracked/vidur/data/profiling/network/<network_device>/`

It does not call `extern/tracked/vidur/vidur/profiling/collectives/main.py` (Vidur’s Ray-based collectives profiler) to
generate host-specific CSVs.

## Mitigations / Workarounds

- If you only need single-GPU sims: keep `tensor_parallel_size=1` and `pipeline_parallel_size=1`.
- If you need TP/PP modeling:
  - Manually run Vidur’s collectives profiler on your host and stage `all_reduce.csv` / `send_recv.csv` into your
    profiling root. See `context/summaries/vidur-kb/howto-profile-vidur-network-collectives.md`.
  - Ensure `hardware.network_device` matches the directory you staged.

## What “fixed” would look like

- Add an explicit “network profiling” stage that runs Vidur’s collectives profilers (all-reduce + send/recv) on the
  current host (and optionally multi-node), then writes into:
  `data/profiling/network/<network_device>/{all_reduce,send_recv}.csv`.
- Wire that into `vidur-profile` and `vidur-cli svr profile` so the profiling root is self-contained and host-specific.

