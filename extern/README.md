# External code (`extern/`)

`extern/` contains third-party source code used by this repository (typically simulators and
supporting tools). We track these dependencies as **git submodules** so the main repo stays
lightweight and reproducible (pinned to specific commits via `.gitmodules`).

This directory is for *code*. Large machine-local assets (models/datasets) live under `models/`
and `datasets/`.

## Contents

Submodules live under `extern/tracked/`:

- `extern/tracked/vidur/`: Vidur (Microsoft) — https://github.com/microsoft/vidur
- `extern/tracked/LLMCompass/`: LLMCompass (Princeton) — https://github.com/PrincetonUniversity/LLMCompass

## Clone / Setup

```bash
git submodule update --init --recursive
```

If you see an uninitialized submodule or a changed URL:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

## Updating a submodule

Submodules are pinned intentionally. To update one, `cd` into the submodule, fetch, checkout the
desired commit/tag/branch, then commit the updated submodule pointer in the parent repo.

Avoid committing local modifications inside `extern/*`; prefer updating the pinned commit or
carrying changes as patches in the main repo.
