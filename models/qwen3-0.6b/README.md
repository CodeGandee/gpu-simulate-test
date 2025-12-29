# Qwen3-0.6B (external reference)

This folder manages a machine-local reference to the Qwen3-0.6B model files.

## Layout

- `source-data` (symlink, ignored by git): points at the local directory containing `config.json`,
  `tokenizer.json`, and `model.safetensors`.
- `bootstrap.sh`: recreates/repairs the `source-data` symlink.

## Bootstrap

```bash
# Option A (recommended): point EXTERNAL_REF_ROOT at your model storage root
export EXTERNAL_REF_ROOT=/path/to/llm-models
bash models/qwen3-0.6b/bootstrap.sh

# Option B: rely on the repository's detected default path (if it exists on this machine)
bash models/qwen3-0.6b/bootstrap.sh
```

