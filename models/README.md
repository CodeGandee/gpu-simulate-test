# External references (models)

This directory holds references to large, third-party, and/or machine-local model assets.
Only small metadata and bootstrap scripts are committed; the actual model data is kept out of git.

Managed references:

- `qwen3-0.6b/` — Qwen3 0.6B model files (local external storage)
  - `source-data`: symlink to the local model directory (ignored by git)
  - Bootstrap: `bash models/qwen3-0.6b/bootstrap.sh`
- `llama2-7b-hf/` — LLaMA2-7B (`meta-llama/Llama-2-7b-hf`)
  - `source-data`: symlink to the local model directory (ignored by git)
  - Bootstrap: `bash models/llama2-7b-hf/bootstrap.sh`
- `llama2-70b-hf/` — LLaMA2-70B (`meta-llama/Llama-2-70b-hf`)
  - `source-data`: symlink to the local model directory (ignored by git)
  - Bootstrap: `bash models/llama2-70b-hf/bootstrap.sh`
- `internlm-20b/` — InternLM-20B (`internlm/internlm-20b`)
  - `source-data`: symlink to the local model directory (ignored by git)
  - Bootstrap: `bash models/internlm-20b/bootstrap.sh`
- `qwen-72b/` — Qwen-72B (`Qwen/Qwen-72B`)
  - `source-data`: symlink to the local model directory (ignored by git)
  - Bootstrap: `bash models/qwen-72b/bootstrap.sh`

Populate/repair all model references:

```bash
bash models/bootstrap.sh
```

If you don't have a given model downloaded locally yet, its bootstrap will print a warning; run the per-model bootstrap after placing the files under your `GSIM_MODELS_ROOT` (legacy: `EXTERNAL_REF_ROOT`).

Common options:
- `--yes`: non-interactive mode (auto-replace links).
- `--clean`: remove repo symlinks only (does not delete model data).
