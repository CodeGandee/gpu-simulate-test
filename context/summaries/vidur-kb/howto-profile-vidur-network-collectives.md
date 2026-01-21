# Howto: profile Vidur network collectives and use them in runs

This note covers how to generate *your own* Vidur network profiling CSVs (`all_reduce.csv`, `send_recv.csv`) for a
specific host topology (NVLink vs PCIe, multi-node, etc), and how to use them in this repo’s `vidur-sim` / `vidur-cli`
workflows.

Sources:
- Vidur upstream profiling docs: https://github.com/microsoft/vidur/blob/main/docs/profiling.md
- This repo stages/reads the network CSVs under a profiling root as:
  - `data/profiling/network/<network_device>/all_reduce.csv`
  - `data/profiling/network/<network_device>/send_recv.csv`
  (see `src/gpu_simulate_test/vidur_ext/sim_runner.py`)

## When you need network profiling

- If `tensor_parallel_size > 1`, Vidur loads `all_reduce.csv` and models TP collective time.
- If `pipeline_parallel_size/num_pipeline_stages > 1`, Vidur loads `send_recv.csv` and models PP communication time.
- If both are 1, network collectives are not used.

## Pick a `network_device` id

`network_device` is just an identifier used to pick the directory under `data/profiling/network/<network_device>/`.

Choose a new name that describes your host, for example:
- `myhost_8xa100_nvlink`
- `myhost_4xh100_pcie`

You must use the same `network_device` value consistently:
- while generating/staging network CSVs into the profiling root
- in the configs used by `vidur-sim` / `vidur-cli` (via `hardware.network_device=...`)

## Profile network collectives (Vidur collectives profiler)

Vidur’s collectives profiler is a Ray-based benchmark runner:
- Entry point: `extern/tracked/vidur/vidur/profiling/collectives/main.py`
- Outputs: `.../all_reduce.csv` and `.../send_recv.csv`

Example (single node, 8 GPUs):

```bash
# Optional: pin which GPUs Ray should see
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Choose where raw profiling outputs land
OUT_DIR="tmp/vidur_collectives_profile"

# Profile all-reduce (needed for TP>1)
pixi run python extern/tracked/vidur/vidur/profiling/collectives/main.py \
  --collective all_reduce \
  --num_workers_per_node_combinations 1 2 4 8 \
  --output_dir "${OUT_DIR}"

# Profile send/recv (needed for PP>1)
pixi run python extern/tracked/vidur/vidur/profiling/collectives/main.py \
  --collective send_recv \
  --num_workers_per_node_combinations 1 2 4 8 \
  --output_dir "${OUT_DIR}"
```

The profiler writes into a timestamped directory like:
- `${OUT_DIR}/collective/<timestamp>/all_reduce.csv`
- `${OUT_DIR}/collective/<timestamp>/send_recv.csv`

## Stage the CSVs into a profiling root

Vidur simulations in this repo expect the CSVs inside the *profiling root*:

```text
<profiling_root>/
  data/profiling/
    network/<network_device>/
      all_reduce.csv
      send_recv.csv
```

Example copy:

```bash
PROFILING_ROOT="/abs/path/to/<run_dir>/profile"
NETWORK_DEVICE="myhost_8xa100_nvlink"
TS_DIR="$(ls -1 "${OUT_DIR}/collective" | tail -n 1)"

mkdir -p "${PROFILING_ROOT}/data/profiling/network/${NETWORK_DEVICE}"
cp "${OUT_DIR}/collective/${TS_DIR}/all_reduce.csv" \
  "${PROFILING_ROOT}/data/profiling/network/${NETWORK_DEVICE}/all_reduce.csv"
cp "${OUT_DIR}/collective/${TS_DIR}/send_recv.csv" \
  "${PROFILING_ROOT}/data/profiling/network/${NETWORK_DEVICE}/send_recv.csv"
```

## Use the profiles in actual runs

1) Ensure your run config sets `hardware.network_device=<network_device>`.
   - In Hydra terms, you can override it directly:

```bash
pixi run vidur-cli svr profile \
  hardware.network_device=myhost_8xa100_nvlink \
  ...
```

2) Ensure your simulation uses the same profiling root and the same `hardware.network_device`.
   - If TP/PP > 1 and the CSVs are missing, Vidur will error when trying to train the network models.

## Current limitation in this repo

This repo’s current `vidur-profile` / `vidur-cli ... profile` flows do not *generate* network profiles on your host.
They only *copy* vendor-provided CSVs when `profiling.include_network=true` and when the requested
`hardware.network_device` exists under `extern/tracked/vidur/data/profiling/network/`.

If you want host-specific network profiles today, you must run the collectives profiler manually and stage the CSVs
yourself (as shown above).

