from __future__ import annotations

import hashlib
from typing import Any

from omegaconf import OmegaConf


def _stable_id(prefix: str, *parts: Any, length: int = 10) -> str:
    normalized = [prefix] + [str(p) for p in parts]
    payload = "\x1f".join(normalized).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:length]
    return f"{prefix}_{digest}"


def register_omegaconf_resolvers() -> None:
    # Register once (Hydra may import entrypoints multiple times in tests).
    if OmegaConf.has_resolver("stable_id"):
        return
    OmegaConf.register_new_resolver("stable_id", _stable_id)

