# LLaMA2-70B (external reference)

This folder manages a machine-local reference to the `meta-llama/Llama-2-70b-hf` model files (Vidur paper model set).

## Layout

- `source-data` (symlink, ignored by git): points at the local directory containing `config.json`,
  tokenizer files, and `*.safetensors` shards (or equivalent).
- `bootstrap.sh`: recreates/repairs the `source-data` symlink.

## Bootstrap

```bash
# Option A (recommended): point GSIM_MODELS_ROOT at your model storage root
export GSIM_MODELS_ROOT=/path/to/llm-models
bash models/llama2-70b-hf/bootstrap.sh

# Option B: rely on the repository's detected default path (if it exists on this machine)
bash models/llama2-70b-hf/bootstrap.sh
```
