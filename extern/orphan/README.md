# Orphan checkouts

`extern/orphan/` is for local, non-submodule checkouts of large third-party repos (for quick patching, grepping, or experimentation).

- Everything under `extern/orphan/*/` is git-ignored.
- Files directly under `extern/orphan/` (like this README) are tracked so contributors understand the convention.

Current convention:

- `extern/orphan/vllm/`: shallow clone of https://github.com/vllm-project/vllm (ignored)

