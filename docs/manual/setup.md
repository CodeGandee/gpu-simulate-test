# Setup

## 1) Initialize the repo

```bash
git submodule update --init --recursive
pixi install
```

For more environment details, see `context/instructions/prep-dev-env.md`.

## 2) GPU requirements (for real-bench + profiling)

- An NVIDIA GPU is required for:
  - `pixi run real-bench ...` when `hardware.device=cuda:0`
  - `pixi run vidur-profile ...` (profiling captures GPU kernel timing)
  - `pixi run paper-fidelity repro ...` (Sarathi real replay)
- Quick sanity check:

```bash
pixi run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## 3) Model references (external symlinks)

This repo uses an “external reference” pattern: model weights are stored outside git, and the repo keeps a symlink under `models/`.

```bash
bash models/bootstrap.sh
```

Or bootstrap only the model(s) you need:

- Compare workflow (`001-compare-vidur-real-timing`): `bash models/qwen3-0.6b/bootstrap.sh`
- Paper fidelity baseline (`002-reproduce-vidur-paper-fidelity`): `bash models/llama2-7b-hf/bootstrap.sh`

The model symlink (e.g. `models/qwen3-0.6b/source-data -> /path/to/Qwen3-0.6B`) should contain at least:

- `config.json`
- `tokenizer.json` or `tokenizer.model`
- `model.safetensors` (or equivalent)
