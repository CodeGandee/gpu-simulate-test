# Setup

## 1) Initialize the repo

```bash
git submodule update --init --recursive
pixi install
```

## 2) GPU requirements (for real-bench + profiling)

- An NVIDIA GPU is required for:
  - `pixi run real-bench ...` when `hardware.device=cuda:0`
  - `pixi run vidur-profile ...` (profiling captures GPU kernel timing)
- Quick sanity check:

```bash
pixi run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## 3) Model reference (Qwen3-0.6B)

This repo uses an “external reference” pattern: the model weights are stored outside git, and the repo keeps a symlink under `models/`.

```bash
bash models/qwen3-0.6b/bootstrap.sh
```

This creates/repairs `models/qwen3-0.6b/source-data -> /path/to/Qwen3-0.6B` and should contain at least:

- `config.json`
- `tokenizer.json` or `tokenizer.model`
- `model.safetensors` (or equivalent)

