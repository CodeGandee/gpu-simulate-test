# Qwen-72B (external reference)

This folder manages a machine-local reference to the `Qwen/Qwen-72B` model files (Vidur paper model set).

## Layout

- `source-data` (symlink, ignored by git): points at the local directory containing `config.json`,
  tokenizer files, and `*.safetensors` shards (or equivalent).
- `bootstrap.sh`: recreates/repairs the `source-data` symlink.

## Bootstrap

```bash
# Option A (recommended): point GSIM_MODELS_ROOT at your model storage root
export GSIM_MODELS_ROOT=/path/to/llm-models
bash models/qwen-72b/bootstrap.sh

# Option B: rely on the repository's detected default path (if it exists on this machine)
bash models/qwen-72b/bootstrap.sh
```
