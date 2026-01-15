"""
Resource resolution for `vidur-cli`.

This module resolves repo/workspace roots and other filesystem resources with
clear provenance (env vs config TOML vs fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from gpu_simulate_test.io import build_env_snapshot, utcnow_iso, write_json
from gpu_simulate_test.vidur_cli.errors import UserFacingError
from gpu_simulate_test.vidur_cli.search_path import build_config_roots


ResolvedSource = Literal["env", "config_toml", "repo_fallback", "pwd_default"]
ConfigTomlSource = Literal["flag", "env", "default"]
WorkspaceSource = Literal["env", "config_toml", "pwd_default"]


@dataclass(frozen=True)
class ResolvedPath:
    value: Path
    source: ResolvedSource
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResolvedWorkspace:
    value: Path
    source: WorkspaceSource
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResolvedConfigToml:
    path: Path
    source: ConfigTomlSource


@dataclass(frozen=True)
class ResourceMapV1:
    repo_root: ResolvedPath
    models_root: ResolvedPath
    datasets_root: ResolvedPath
    workspace_root: ResolvedWorkspace
    config_toml: ResolvedConfigToml
    hydra_config_roots: list[Path]

    def to_json(self) -> Mapping[str, Any]:
        env_snapshot = dict(build_env_snapshot())
        for key in [
            "GSIM_REPO_ROOT",
            "GSIM_MODELS_ROOT",
            "GSIM_DATASETS_ROOT",
            "GSIM_VIDUR_WORKSPACE_DIR",
            "GSIM_VIDUR_CLI_HYDRA_CONFIG_DIRS",
            "GSIM_VIDUR_CLI_USER_CONFIG",
        ]:
            env_snapshot[key] = os.environ.get(key)

        return {
            "schema_version": "v1",
            "resolved_at": utcnow_iso(),
            "repo_root": _resolved_path_json(self.repo_root),
            "models_root": _resolved_path_json(self.models_root),
            "datasets_root": _resolved_path_json(self.datasets_root),
            "workspace_root": _resolved_workspace_json(self.workspace_root),
            "config_toml": {
                "path": str(self.config_toml.path.expanduser().resolve()),
                "source": self.config_toml.source,
            },
            "hydra_config_roots": [str(p.expanduser().resolve()) for p in self.hydra_config_roots],
            "env_snapshot": env_snapshot,
        }


def load_project_config_toml(*, path: Path, required: bool) -> dict[str, Any]:
    """Parse a project-local config TOML.

    Parameters
    ----------
    path
        TOML file path (may be relative; callers should normalize before passing).
    required
        If True, missing files are treated as an error. If False, missing files
        return an empty dict.
    """
    path = path.expanduser()
    if not path.exists():
        if required:
            raise UserFacingError(
                f"Config TOML does not exist: {path}",
                hint="Fix the path or omit --user-config to use the default location.",
            )
        return {}

    try:
        import tomllib  # py3.11+
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError("tomllib is required (Python 3.11+).") from e

    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        raise UserFacingError(
            f"Failed to parse config TOML: {path}",
            hint="Validate the TOML syntax and try again.",
        ) from e

    if not isinstance(data, dict):
        raise UserFacingError(
            f"Config TOML must parse to a table/object, got {type(data).__name__}",
            hint="Use a top-level TOML table (key/value pairs).",
        )
    return dict(data)


def write_resources_json(*, run_dir: Path, resources: ResourceMapV1) -> Path:
    """Write `<run_dir>/resources.json` (schema v1) and return the absolute path."""
    out = run_dir / "resources.json"
    write_json(out, resources.to_json())
    return out.resolve()


def _resolved_path_json(p: ResolvedPath) -> dict[str, Any]:
    out: dict[str, Any] = {"value": str(p.value.expanduser().resolve()), "source": p.source}
    if p.details:
        out["details"] = dict(p.details)
    return out


def _resolved_workspace_json(p: ResolvedWorkspace) -> dict[str, Any]:
    out: dict[str, Any] = {"value": str(p.value.expanduser().resolve()), "source": p.source}
    if p.details:
        out["details"] = dict(p.details)
    return out


def resolve_resources(
    *,
    pwd: Path,
    user_config_flag: str | None,
    cli_config_dirs: Sequence[str],
) -> ResourceMapV1:
    """Resolve resources for a `vidur-cli` invocation."""
    return resolve_resources_from_cli(
        pwd=pwd,
        user_config_flag=user_config_flag,
        cli_config_dirs=cli_config_dirs,
    )


def resolve_resources_from_cli(
    *,
    pwd: Path,
    user_config_flag: str | None,
    cli_config_dirs: Sequence[str],
) -> ResourceMapV1:
    """Resolve resources for a `vidur-cli` invocation."""
    pwd = pwd.expanduser().resolve()

    resolved_config_toml = _resolve_config_toml(pwd=pwd, user_config_flag=user_config_flag)
    toml = load_project_config_toml(
        path=resolved_config_toml.path, required=resolved_config_toml.source in {"flag", "env"}
    )

    resolved_repo_root = _resolve_repo_root(pwd=pwd, toml=toml)
    _validate_repo_root(resolved_repo_root.value, pwd=pwd, config_toml=resolved_config_toml)

    resolved_models_root = _resolve_models_root(repo_root=resolved_repo_root.value, pwd=pwd, toml=toml)
    resolved_datasets_root = _resolve_datasets_root(repo_root=resolved_repo_root.value, pwd=pwd, toml=toml)
    _validate_required_dir(
        resolved_models_root.value,
        what="models_root",
        hint="Set GSIM_MODELS_ROOT or add resources.models_root to your config TOML.",
        context={"resolved": _resolved_path_json(resolved_models_root)},
    )
    _validate_required_dir(
        resolved_datasets_root.value,
        what="datasets_root",
        hint="Set GSIM_DATASETS_ROOT or add resources.datasets_root to your config TOML.",
        context={"resolved": _resolved_path_json(resolved_datasets_root)},
    )

    resolved_workspace_root = _resolve_workspace_root(pwd=pwd, toml=toml)

    toml_config_dirs = _get_toml_str_list(toml, ("hydra", "config_dirs"))
    hydra_config_roots = build_config_roots(
        repo_root=resolved_repo_root.value,
        pwd=pwd,
        cli_config_dirs=cli_config_dirs,
        env_config_dirs=os.environ.get("GSIM_VIDUR_CLI_HYDRA_CONFIG_DIRS"),
        toml_config_dirs=toml_config_dirs,
    )
    missing_dirs = [str(p) for p in hydra_config_roots if not p.exists()]
    if missing_dirs:
        raise UserFacingError(
            "One or more Hydra config roots do not exist.",
            hint="Fix your --config-dir / env / TOML hydra.config_dirs entries or remove them.",
            context={"missing_config_roots": missing_dirs, "hydra_config_roots": [str(p) for p in hydra_config_roots]},
        )

    return ResourceMapV1(
        repo_root=resolved_repo_root,
        models_root=resolved_models_root,
        datasets_root=resolved_datasets_root,
        workspace_root=resolved_workspace_root,
        config_toml=resolved_config_toml,
        hydra_config_roots=hydra_config_roots,
    )


def _resolve_config_toml(*, pwd: Path, user_config_flag: str | None) -> ResolvedConfigToml:
    attempted: list[dict[str, Any]] = []

    if user_config_flag is not None:
        raw = str(user_config_flag)
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = pwd / p
        attempted.append({"source": "flag", "path": str(p)})
        return ResolvedConfigToml(path=p.resolve(), source="flag")

    env_path = os.environ.get("GSIM_VIDUR_CLI_USER_CONFIG")
    if env_path:
        p = Path(env_path).expanduser()
        if not p.is_absolute():
            p = pwd / p
        attempted.append({"source": "env", "env_var": "GSIM_VIDUR_CLI_USER_CONFIG", "path": str(p)})
        return ResolvedConfigToml(path=p.resolve(), source="env")

    default_path = (pwd / ".vidur-config" / "default.toml").resolve()
    attempted.append({"source": "default", "path": str(default_path)})
    return ResolvedConfigToml(path=default_path, source="default")


def _resolve_repo_root(*, pwd: Path, toml: Mapping[str, Any]) -> ResolvedPath:
    env_val = os.environ.get("GSIM_REPO_ROOT")
    attempted: list[dict[str, Any]] = [{"source": "env", "env_var": "GSIM_REPO_ROOT", "value": env_val}]
    if env_val:
        return ResolvedPath(value=Path(env_val).expanduser().resolve(), source="env", details={"env_var": "GSIM_REPO_ROOT"})

    toml_val = _get_toml_str(toml, ("resources", "repo_root"))
    attempted.append({"source": "config_toml", "key": "resources.repo_root", "value": toml_val})
    if toml_val:
        p = Path(toml_val).expanduser()
        if not p.is_absolute():
            p = pwd / p
        return ResolvedPath(
            value=p.resolve(),
            source="config_toml",
            details={"toml_key": "resources.repo_root"},
        )

    attempted.append({"source": "pwd_default", "value": str(pwd)})
    return ResolvedPath(value=pwd, source="pwd_default", details={"pwd": str(pwd)})


def _validate_repo_root(repo_root: Path, *, pwd: Path, config_toml: ResolvedConfigToml) -> None:
    expected = [
        repo_root / "pyproject.toml",
        repo_root / "configs" / "compare_vidur_real",
        repo_root / "src" / "gpu_simulate_test",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if not missing:
        return

    attempted_sources = [
        {"kind": "env", "env_var": "GSIM_REPO_ROOT"},
        {"kind": "config_toml", "key": "resources.repo_root", "config_toml": str(config_toml.path)},
        {"kind": "pwd_default", "pwd": str(pwd)},
    ]
    raise UserFacingError(
        f"repo_root does not look like the gpu-simulate-test repository: {repo_root}",
        hint="Set GSIM_REPO_ROOT to the repo root (the directory containing pyproject.toml).",
        context={"missing_expected_paths": missing, "attempted_sources": attempted_sources},
    )


def _resolve_models_root(*, repo_root: Path, pwd: Path, toml: Mapping[str, Any]) -> ResolvedPath:
    env_val = os.environ.get("GSIM_MODELS_ROOT")
    if env_val:
        return ResolvedPath(
            value=_resolve_path_like(env_val, pwd=pwd),
            source="env",
            details={"env_var": "GSIM_MODELS_ROOT"},
        )

    toml_val = _get_toml_str(toml, ("resources", "models_root"))
    if toml_val:
        return ResolvedPath(
            value=_resolve_path_like(toml_val, pwd=pwd),
            source="config_toml",
            details={"toml_key": "resources.models_root"},
        )

    return ResolvedPath(value=(repo_root / "models").resolve(), source="repo_fallback", details={"repo_fallback": True})


def _resolve_datasets_root(*, repo_root: Path, pwd: Path, toml: Mapping[str, Any]) -> ResolvedPath:
    env_val = os.environ.get("GSIM_DATASETS_ROOT")
    if env_val:
        return ResolvedPath(
            value=_resolve_path_like(env_val, pwd=pwd),
            source="env",
            details={"env_var": "GSIM_DATASETS_ROOT"},
        )

    toml_val = _get_toml_str(toml, ("resources", "datasets_root"))
    if toml_val:
        return ResolvedPath(
            value=_resolve_path_like(toml_val, pwd=pwd),
            source="config_toml",
            details={"toml_key": "resources.datasets_root"},
        )

    return ResolvedPath(
        value=(repo_root / "datasets").resolve(), source="repo_fallback", details={"repo_fallback": True}
    )


def _resolve_workspace_root(*, pwd: Path, toml: Mapping[str, Any]) -> ResolvedWorkspace:
    env_val = os.environ.get("GSIM_VIDUR_WORKSPACE_DIR")
    if env_val:
        return _workspace_from_dir_value(env_val, pwd=pwd, source="env", details={"env_var": "GSIM_VIDUR_WORKSPACE_DIR"})

    toml_val = _get_toml_str(toml, ("resources", "workspace_dir"))
    if toml_val:
        return _workspace_from_dir_value(toml_val, pwd=pwd, source="config_toml", details={"toml_key": "resources.workspace_dir"})

    return _workspace_from_dir_value("default", pwd=pwd, source="pwd_default", details={"default": True})


def _workspace_from_dir_value(
    value: str,
    *,
    pwd: Path,
    source: WorkspaceSource,
    details: Mapping[str, Any],
) -> ResolvedWorkspace:
    value = value.strip()
    if not value:
        raise UserFacingError("workspace_dir must be non-empty.", hint="Set GSIM_VIDUR_WORKSPACE_DIR=default (or another name).")
    p = Path(value).expanduser()
    if p.is_absolute():
        return ResolvedWorkspace(value=p.resolve(), source=source, details=dict(details))
    return ResolvedWorkspace(value=(pwd / ".vidur-output" / p).resolve(), source=source, details=dict(details))


def _resolve_path_like(raw: str, *, pwd: Path) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = pwd / p
    return p.resolve()


def _validate_required_dir(path: Path, *, what: str, hint: str, context: Mapping[str, Any]) -> None:
    if path.exists() and path.is_dir():
        return
    raise UserFacingError(f"{what} does not exist or is not a directory: {path}", hint=hint, context=dict(context))


def _get_toml_str(toml: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    cur: Any = toml
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if cur is None:
        return None
    return str(cur)


def _get_toml_str_list(toml: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    cur: Any = toml
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return []
        cur = cur[key]
    if cur is None:
        return []
    if isinstance(cur, list):
        return [str(x) for x in cur]
    return [str(cur)]
