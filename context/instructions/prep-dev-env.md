# Prep: development environment

This repo expects developers to use the Pixi-managed environment and treat large model assets as
machine-local (not committed to git).

## 1) Use the Pixi environment (don’t use system Python)

- Create/update the dev environment:
  - `pixi install`
- Run commands inside the environment (recommended):
  - `pixi run python -c "import torch; print(torch.__version__)"`
  - `pixi run python -m pip list`
- Or drop into a shell that has the right Python on `PATH`:
  - `pixi shell`

Avoid running `python`, `pip`, or `pytest` from your system environment; use `pixi run ...` instead.

## 2) External source repos under `extern/`

This repo vendors or tracks third-party source trees under `extern/` so you can inspect upstream
implementations when you need to understand APIs, configs, or simulator assumptions.

Initialize submodules after cloning:

- `git submodule update --init --recursive`

Tracked submodules:

- `extern/tracked/vidur/`: Vidur simulator
- `extern/tracked/sarathi-serve/`: Sarathi-Serve (LLM serving engine; also useful as a reference)
- `extern/tracked/LLMCompass/`: LLMCompass (design space exploration / cost modeling)

Other vendored source:

- `extern/orphan/vllm/`: vLLM snapshot/fork used for reference (not necessarily wired into the
  project as an installed dependency)

If you’re unsure how to use a library integrated by this repo, prefer reading the source under
`extern/` (and its README/examples) before adding new dependencies.

## 3) What’s in `models/`

`models/` uses an “external reference” pattern:

- Small metadata, READMEs, and bootstrap scripts are committed to git.
- The actual model weights/tokenizers live in machine-local storage and are linked in via symlinks
  (ignored by git).

Key entrypoints:

- Bootstrap all model references: `bash models/bootstrap.sh`
- Example reference: `models/qwen3-0.6b/`
  - `models/qwen3-0.6b/source-data` is a symlink to your local model directory.
  - `bash models/qwen3-0.6b/bootstrap.sh` recreates/repairs the symlink.
  - Supports `EXTERNAL_REF_ROOT=/path/to/llm-models` to point at your own storage root.

