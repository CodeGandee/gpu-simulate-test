# External references (models)

This directory holds references to large, third-party, and/or machine-local model assets.
Only small metadata and bootstrap scripts are committed; the actual model data is kept out of git.

Managed references:

- `qwen3-0.6b/` — Qwen3 0.6B model files (local external storage)
  - `source-data`: symlink to the local model directory (ignored by git)
  - Bootstrap: `bash models/qwen3-0.6b/bootstrap.sh`

Populate/repair all model references:

```bash
bash models/bootstrap.sh
```

