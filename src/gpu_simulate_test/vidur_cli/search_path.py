"""
Hydra config search path helpers for `vidur-cli`.

This module centralizes:
- config-root precedence and normalization
- filesystem-only discovery of config groups and preset keys
- Hydra programmatic composition based on resolved config roots
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from gpu_simulate_test.config import register_omegaconf_resolvers
from gpu_simulate_test.vidur_cli.errors import UserFacingError


def build_config_roots(
    *,
    repo_root: Path,
    pwd: Path,
    cli_config_dirs: Iterable[str],
    env_config_dirs: str | None,
    toml_config_dirs: Iterable[str],
) -> list[Path]:
    """Resolve Hydra config roots with the precedence defined in the spec."""
    roots: list[Path] = []
    for raw in list(cli_config_dirs) + _split_env_dirs(env_config_dirs) + list(toml_config_dirs):
        roots.append(_normalize_dir(raw, pwd=pwd))
    roots.append((repo_root / "configs" / "compare_vidur_real").expanduser().resolve())
    return _dedupe_paths(roots)


def _normalize_dir(raw: str, *, pwd: Path) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = pwd / p
    return p.resolve()


def _split_env_dirs(value: str | None) -> list[str]:
    if not value:
        return []
    return [p for p in value.split(os.pathsep) if p]


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


@dataclass(frozen=True)
class PresetEntry:
    """A discovered preset key for a group, including override provenance."""

    group: str
    key: str
    active_path: Path
    all_paths: list[Path]


def discover_groups(config_roots: list[Path]) -> list[str]:
    """Return sorted unique group names across all config roots."""
    groups: set[str] = set()
    for root in config_roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir():
                groups.add(child.name)
    return sorted(groups)


def list_presets_for_group(*, group: str, config_roots: list[Path]) -> list[PresetEntry]:
    """Return preset keys and their source paths (first root wins)."""
    hits: dict[str, list[Path]] = {}
    for root in config_roots:
        d = root / group
        if not d.exists() or not d.is_dir():
            continue
        for y in sorted(d.glob("*.yaml")):
            hits.setdefault(y.stem, []).append(y.resolve())

    out: list[PresetEntry] = []
    for key, paths in sorted(hits.items(), key=lambda kv: kv[0]):
        out.append(PresetEntry(group=group, key=key, active_path=paths[0], all_paths=paths))
    return out


def compose_config(
    *,
    config_name: str,
    config_roots: Sequence[Path],
    overrides: Sequence[str],
) -> object:
    """Compose a Hydra config using the resolved config roots.

    Notes
    -----
    Hydra's primary config directory has higher precedence than `hydra.searchpath`.
    To allow user-provided config dirs to override in-repo group presets, this helper
    copies the chosen primary config YAML into an isolated temporary directory and
    relies on `hydra.searchpath` for group/preset resolution.
    """
    register_omegaconf_resolvers()

    primary_cfg = _find_primary_config(config_name=config_name, config_roots=config_roots)
    searchpath = [f"file://{p.expanduser().resolve()}" for p in config_roots]

    try:
        from hydra import compose, initialize_config_dir  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("hydra-core is required; run inside the Pixi env (`pixi install`).") from e

    with tempfile.TemporaryDirectory(prefix="vidur-cli-hydra-") as tmp:
        tmp_dir = Path(tmp)
        tmp_primary = tmp_dir / primary_cfg.name
        tmp_primary.write_text(primary_cfg.read_text(encoding="utf-8"), encoding="utf-8")

        with initialize_config_dir(config_dir=str(tmp_dir), job_name="vidur_cli", version_base=None):
            cfg = compose(
                config_name=config_name,
                overrides=[f"hydra.searchpath=[{','.join(searchpath)}]", *list(overrides)],
            )
            return cfg


def _find_primary_config(*, config_name: str, config_roots: Sequence[Path]) -> Path:
    candidates = [f"{config_name}.yaml", f"{config_name}.yml"]
    for root in config_roots:
        for name in candidates:
            p = (root / name).expanduser()
            if p.exists():
                return p.resolve()
    raise UserFacingError(
        f"Cannot find primary config {config_name!r} in any config root.",
        hint="Check your config roots (--config-dir / env / TOML) or use an in-repo config.",
        context={"config_name": config_name, "config_roots": [str(p) for p in config_roots]},
    )

