# Hydra configs

All user-facing commands are Hydra apps with presets under `configs/compare_vidur_real/`.

## Stage configs

- Workload generation: `configs/compare_vidur_real/workload_spec.yaml`
- Real timing: `configs/compare_vidur_real/real_bench.yaml`
- Vidur profiling: `configs/compare_vidur_real/vidur_profile.yaml`
- Vidur sim: `configs/compare_vidur_real/vidur_sim.yaml`
- Comparison: `configs/compare_vidur_real/compare_runs.yaml`

Each stage sets `hydra.run.dir` so the outputs land under `tmp/.../<stable_id>/`.

## Config groups

- `configs/compare_vidur_real/model/` (e.g. `qwen3_0_6b.yaml`)
- `configs/compare_vidur_real/hardware/` (e.g. `a100.yaml`)
- `configs/compare_vidur_real/workload/` (e.g. `default.yaml`)
- `configs/compare_vidur_real/backend/` (`transformers.yaml`, `sarathi.yaml`)
- `configs/compare_vidur_real/vidur/` (profiling root + model key)

## Common overrides

```bash
# Change prompts input + decode length
pixi run workload-spec \
  workload.prompts=tmp/prompts/example.prompts.jsonl \
  workload.num_decode_tokens=64 \
  workload.seed=123

# Run real-bench with Sarathi and load weights from local path
CUDA_VISIBLE_DEVICES=0 pixi run real-bench \
  backend=sarathi \
  model.model_id=$(pwd)/models/qwen3-0.6b/source-data \
  workload.workload_dir=tmp/workloads/<workload_id>

# Point Vidur to an explicit profiling root
pixi run vidur-sim \
  vidur.profiling.root=tmp/vidur_profiling/a100/qwen3_0_6b \
  workload.workload_dir=tmp/workloads/<workload_id>
```

