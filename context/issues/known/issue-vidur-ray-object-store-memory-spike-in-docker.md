# Known issue: Vidur/Sarathi Ray object store reserves huge `/dev/shm` (looks like “all RAM”)

## Summary

When running:

`docs/tutorial/howto/tut-sim-vs-real-with-vidur-cli/run_demo_static_from_pf_trace.sh --snapshot-report`

**inside a Docker container**, the run can appear to “use all system memory” (and may get OOM-killed). The
`--snapshot-report` flag is a red herring: the spike happens earlier (typically during `svr profile` and/or
`svr real`) when Vidur/Sarathi start **Ray**, and Ray’s **default object store** is large on this machine.

## What’s happening (root cause)

Both **Vidur profiling** and **Sarathi** use Ray:

- Vidur profiling MLP uses `ray.remote(...)` actors (auto-inits Ray if not already running):
  - `extern/tracked/vidur/vidur/profiling/mlp/main.py`
- Sarathi always initializes a Ray cluster when an engine is created:
  - `extern/tracked/sarathi-serve/sarathi/engine/base_llm_engine.py` → `initialize_cluster()` → `ray.init(...)`

On this host, Ray defaults to a **large object store** backed by **tmpfs** (`/dev/shm`), which is effectively RAM.

Ray 2.53.0 defaults (see `ray._private.ray_constants`):

- `DEFAULT_OBJECT_STORE_MEMORY_PROPORTION = 0.3`
- `DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES = 200_000_000_000` (200 GB cap)

And Ray detects:

- `system_memory ≈ 1008 GiB` (`ray._private.utils.get_system_memory()` / `ray._common.utils.get_system_memory()`)
- `/dev/shm size ≈ 504 GiB`

So Ray’s default object store becomes:

`min(0.3 * system_memory, 200GB) = 200GB`, placed under `/dev/shm` (because `/dev/shm` is large enough).

If multiple Ray instances get started (e.g., separate profiling subprocesses, retries, or crashy runs that don’t
clean up), `/dev/shm` usage can balloon and looks like “all system RAM is consumed”.

## Why it shows up “in Docker”

The container `vidur_lbj_test` (`00e6c77f7de5...`) has:

- **No cgroup memory limit** (`HostConfig.Memory=0`). In Docker, `--memory` is implemented via cgroups; a value of
  `0` means “unlimited” (i.e., no container cap). Ray therefore sizes itself against the full host memory (≈ 1TB).
- `--ipc=host` (`HostConfig.IpcMode=host`), so `/dev/shm` inside the container is the host’s ~504GiB tmpfs.

This combination makes Ray’s default `/dev/shm` object store large and fully eligible to allocate.

## “Same code, why only in container?”

The repo code does **not** set Ray object store size; both Vidur and Sarathi call `ray.init(...)` without an
`object_store_memory=` argument. The object store size is therefore decided by **Ray defaults + environment**.

Reasons you may observe the “all memory” behavior in the container but not in your host run:

- **Cgroup limits differ:** Ray uses cgroup memory limits when present. The container has no memory limit
  (`HostConfig.Memory=0`), so Ray sizes against the full host RAM. On the host you might be in a cgroup-limited
  session/job (e.g., Slurm/systemd slice), which would make Ray choose a smaller object store.
- **Different `/dev/shm` setup:** This container uses `--ipc=host`, making `/dev/shm` huge; Ray will happily place a
  large object store there. A “normal” Docker container with the default small `/dev/shm` would either error or
  fall back to disk (if `RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1`), which avoids a big tmpfs allocation.
- **Leaked Ray clusters from aborted runs:** If runs are interrupted (OOM-kill / `kill -9` / crash), Ray’s
  `raylet`/`plasma_store` processes can be left behind. Re-running starts another cluster, and the `/dev/shm`
  allocations add up quickly.
- **Different shell env:** If your host shell sets `RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES` /
  `RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION` (or you start Ray manually with `--object-store-memory`), but the
  container doesn’t inherit those env vars, you’ll see different behavior even with the same code.

## How to confirm quickly

Inside the environment that runs Vidur/Sarathi (Pixi env):

```bash
python - <<'PY'
import os
import ray
from ray._private import ray_constants
from ray._private import utils

st = os.statvfs("/dev/shm")
shm_bytes = st.f_frsize * st.f_blocks
system_mem = utils.get_system_memory()

cap = ray_constants.DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES
prop = ray_constants.DEFAULT_OBJECT_STORE_MEMORY_PROPORTION
default_objstore = int(min(system_mem * prop, cap))

print("ray_version", ray.__version__)
print("system_memory_bytes", system_mem)
print("shm_bytes", shm_bytes)
print("DEFAULT_OBJECT_STORE_MEMORY_PROPORTION", prop)
print("DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES", cap)
print("default_object_store_bytes", default_objstore)
PY
```

If a run is “stuck” after a crash, also check for leftover Ray processes:

```bash
pgrep -af 'raylet|plasma_store' || true
```

## Mitigations / workarounds

### 1) Cap Ray’s default object store via `vidur-cli` config (recommended)

This repo supports configuring a small set of Ray runtime settings via Hydra config under `ray.env.*`
with per-setting precedence:

1. Environment (`RAY_*` already set)
2. Configuration (`ray.env.*`)
3. Ray defaults (no injection)

Example (no manual `export RAY_*` required):

```bash
pixi run vidur-cli svr profile --run-dir <run_dir> \
  profiling.mlp.profile_method=cuda_event \
  ray.env.RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.10 \
  ray.env.RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=20000000000 \
  ray.env.RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=true
```

Notes:
- Defaults are opt-in: the repo default config leaves these values `null`.
- For Ray-backed stages, `vidur-cli` emits an effective settings report to stderr and writes
  `<run_dir>/<stage>/ray_settings.json` for reproducibility.

### 1.1) (Power-user) Cap Ray via environment variables

If you prefer to set env vars directly, they take precedence over config:

Before running `vidur-cli` stages that use Ray (profiling + Sarathi), set one (or both):

```bash
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=20000000000  # 20 GB
# or
export RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.05        # 5% of system memory
```

This keeps Ray from trying to reserve a ~200GB tmpfs object store.

### 2) Disable Ray in Vidur compute profiling (single-GPU only)

For low-footprint runs, you might expect to disable Ray for Vidur compute profiling:

```bash
pixi run vidur-cli svr profile --run-dir <run_dir> \
  profiling.compute.use_ray=false \
  profiling.mlp.profile_method=cuda_event
```

Current status:

- `profiling.compute.use_ray=false` is **not supported yet** and fails fast in this repo.
- Reason: in the tracked Vidur submodule, the profiling scripts define `--disable_ray` but do not use it
  (they still call `ray.remote(...)` / `ray.get(...)`), and this repo intentionally does not hide missing
  attention profiling data by copying a pre-baked `attention.csv` template.

### 2.1) Status: implemented via feature `006-vidur-cli-ray-config`

Implementation plan: `context/plans/plan-vidur-cli-ray-runtime-config.md`.

### 3) Constrain the container (alternate)

Run the container with an explicit memory limit and/or a smaller shared-memory setup.

Examples:

```bash
# Cap container RAM (and disable extra swap) so Ray sees less "system memory":
docker run ... --memory=256g --memory-swap=256g ...

# Reduce /dev/shm (requires NOT using --ipc=host):
docker run ... --ipc=private --shm-size=64g ...

# Also safe to set Ray caps explicitly via env vars:
docker run ... -e RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=20000000000 ...
```

Notes / caveats:

- Too small `/dev/shm` can make Ray error unless `RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1` is set (falls back to disk).
- `--ipc=host` makes `/dev/shm` shared with the host; `--shm-size` won’t help in that mode.

## Notes

- The `--snapshot-report` flag only copies/sanitizes `<run_dir>/report/` and is not the allocator; Ray is.
- The container’s repo at `/data/FM/libinjun/vidur_hz_2/gpu-simulate-test` is behind this workspace and has local
  modifications, but the memory spike mechanism is primarily Ray defaults + container runtime settings.
