"""
Ray runtime configuration helpers for `vidur-cli`.

This module provides a small, explicitly supported interface for controlling a subset of
Ray runtime settings via environment variables. It is intentionally stdlib-only (no `ray`
imports) so it can be used even when Ray is disabled for compute profiling.

Functions
---------
apply_ray_env_defaults
    Apply Ray env settings with per-key precedence and return a stable report.
write_ray_settings_json
    Persist an effective-settings report as `ray_settings.json`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, Mapping

from gpu_simulate_test.vidur_cli.errors import UserFacingError

RaySettingSource = Literal["environment", "configuration", "default"]


@dataclass(frozen=True)
class RaySetting:
    """A single supported Ray runtime setting with provenance."""

    key: str
    effective_value: str | None
    source: RaySettingSource


SUPPORTED_RAY_ENV_KEYS: tuple[str, ...] = (
    "RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES",
    "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION",
    "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE",
)


def apply_ray_env_defaults(cfg_ray_env: Mapping[str, Any] | None) -> list[RaySetting]:
    """Apply env defaults with env > config > default precedence.

    Notes
    -----
    - If a key is not explicitly set by env or config, `effective_value` remains `None`.
      This avoids computing host-specific derived defaults so reports are comparable across
      host vs Docker runs given the same config.
    """

    cfg_dict = dict(cfg_ray_env or {})
    _validate_no_unknown_keys(cfg_dict)

    out: list[RaySetting] = []
    for key in SUPPORTED_RAY_ENV_KEYS:
        if key in os.environ:
            raw_env_value = os.environ.get(key) or ""
            out.append(
                RaySetting(
                    key=key,
                    effective_value=_serialize_env_value_for_report(key, raw_env_value),
                    source="environment",
                )
            )
            continue

        raw_cfg_value = cfg_dict.get(key)
        if raw_cfg_value is None:
            out.append(RaySetting(key=key, effective_value=None, source="default"))
            continue

        serialized = _serialize_config_value_for_env(key, raw_cfg_value)
        os.environ[key] = serialized
        out.append(RaySetting(key=key, effective_value=serialized, source="configuration"))

    return out


def _validate_no_unknown_keys(cfg_ray_env: Mapping[str, Any]) -> None:
    """Reject unsupported config keys under `ray.env`."""

    unknown = sorted(set(cfg_ray_env) - set(SUPPORTED_RAY_ENV_KEYS))
    if not unknown:
        return
    raise UserFacingError(
        "Unsupported Ray setting key(s) in config (cfg.ray.env).",
        hint="Remove unsupported keys under ray.env.",
        context={"unknown": unknown, "supported": list(SUPPORTED_RAY_ENV_KEYS)},
    )


def _serialize_env_value_for_report(key: str, raw: str) -> str:
    """Validate an existing env var value and return a canonical string for reporting."""

    if key == "RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES":
        return str(_parse_env_int(name=key, raw=raw))
    if key == "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION":
        return _serialize_decimal_for_env(_parse_env_proportion(name=key, raw=raw))
    if key == "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE":
        return "1" if _parse_env_bool(name=key, raw=raw) else "0"
    raise AssertionError(f"Unhandled Ray env key: {key}")


def _serialize_config_value_for_env(key: str, raw: Any) -> str:
    """Validate a config value and serialize it into an env var string."""

    if key == "RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES":
        if not isinstance(raw, int):
            raise UserFacingError(
                "Invalid Ray config value for RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES.",
                hint="Expected a non-negative integer (bytes) or null.",
                context={"value": raw, "type": type(raw).__name__},
            )
        if raw < 0:
            raise UserFacingError(
                "Invalid Ray config value for RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES.",
                hint="Expected a non-negative integer (bytes) or null.",
                context={"value": raw},
            )
        return str(raw)

    if key == "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION":
        if not isinstance(raw, int | float):
            raise UserFacingError(
                "Invalid Ray config value for RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION.",
                hint="Expected a number in (0, 1] or null.",
                context={"value": raw, "type": type(raw).__name__},
            )
        value = float(raw)
        if not (0.0 < value <= 1.0):
            raise UserFacingError(
                "Invalid Ray config value for RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION.",
                hint="Expected a number in (0, 1] or null.",
                context={"value": raw},
            )
        return _serialize_decimal_for_env(Decimal(str(value)))

    if key == "RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE":
        if not isinstance(raw, bool):
            raise UserFacingError(
                "Invalid Ray config value for RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE.",
                hint="Expected a boolean or null.",
                context={"value": raw, "type": type(raw).__name__},
            )
        return "1" if raw else "0"

    raise AssertionError(f"Unhandled Ray env key: {key}")


def _parse_env_int(*, name: str, raw: str) -> int:
    s = raw.strip()
    if not s:
        raise UserFacingError(
            f"Invalid {name} environment value.",
            hint="Expected a non-negative integer string (bytes).",
            context={"value": raw},
        )
    if not s.isdigit():
        raise UserFacingError(
            f"Invalid {name} environment value.",
            hint="Expected a non-negative integer string (bytes).",
            context={"value": raw},
        )
    value = int(s)
    if value < 0:
        raise UserFacingError(
            f"Invalid {name} environment value.",
            hint="Expected a non-negative integer string (bytes).",
            context={"value": raw},
        )
    return value


def _parse_env_proportion(*, name: str, raw: str) -> Decimal:
    s = raw.strip()
    if not s:
        raise UserFacingError(
            f"Invalid {name} environment value.",
            hint="Expected a number in (0, 1].",
            context={"value": raw},
        )
    try:
        value = Decimal(s)
    except (InvalidOperation, ValueError):
        raise UserFacingError(
            f"Invalid {name} environment value.",
            hint="Expected a number in (0, 1].",
            context={"value": raw},
        ) from None
    if not (Decimal("0") < value <= Decimal("1")):
        raise UserFacingError(
            f"Invalid {name} environment value.",
            hint="Expected a number in (0, 1].",
            context={"value": raw},
        )
    return value


def _parse_env_bool(*, name: str, raw: str) -> bool:
    s = raw.strip().lower()
    if s in {"1", "true"}:
        return True
    if s in {"0", "false"}:
        return False
    raise UserFacingError(
        f"Invalid {name} environment value.",
        hint="Expected one of: 1, 0, true, false.",
        context={"value": raw},
    )


def _serialize_decimal_for_env(value: Decimal) -> str:
    """Serialize a Decimal as a stable, non-scientific string."""

    as_fixed = format(value, "f")
    if "." not in as_fixed:
        return as_fixed
    trimmed = as_fixed.rstrip("0").rstrip(".")
    return trimmed or "0"


def write_ray_settings_json(out_path: Path, *, stage: str, settings: list[RaySetting]) -> Path:
    """Write `ray_settings.json` (schema v1) and return an absolute path."""

    payload = {
        "schema_version": "v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": str(stage),
        "settings": [
            {"key": setting.key, "effective_value": setting.effective_value, "source": setting.source}
            for setting in settings
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path.resolve()
