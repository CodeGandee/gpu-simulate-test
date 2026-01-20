"""
Interpreter startup hooks for the Pixi/dev environment.

This module is imported automatically by Python (via `site.py`) when it is
present on `sys.path`. We use it sparingly to apply *lazy* compatibility
patches needed by Ray worker processes, where we cannot rely on a single
driver-side import order.

Currently this repo's tracked Sarathi submodule has an attention wrapper API
shape mismatch with Vidur's attention profiling code:
- Sarathi exposes `sarathi.model_executor.attention.get_attention_wrapper()`
  that calls `FlashinferAttentionWrapper.get_instance()`, but the wrapper
  classes do not define `get_instance()`.
- Vidur expects the returned wrapper instance to implement an "old" interface
  (`init`, `get_cache_block`, `begin_forward`, `forward`, `end_forward`).

The patch below installs a post-import hook that, when (and only when)
`sarathi.model_executor.attention` is imported, injects `get_instance()`
methods that return a per-process singleton implementing the interface Vidur
expects. This keeps the tutorial workflow runnable while still surfacing real
profiling failures (no template fallbacks).
"""

from __future__ import annotations

import importlib.abc
import sys
from importlib.machinery import PathFinder
from types import ModuleType
from typing import Any, Callable


class _PostImportLoader(importlib.abc.Loader):
    """Wrap a real loader and run a callback after module import."""

    def __init__(self, wrapped: importlib.abc.Loader, callback: Callable[[ModuleType], None]) -> None:
        self.m_wrapped = wrapped
        self.m_callback = callback

    def create_module(self, spec):  # type: ignore[no-untyped-def]
        create = getattr(self.m_wrapped, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.m_wrapped.exec_module(module)
        self.m_callback(module)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that patches a specific module after import."""

    def __init__(self, module_name: str, callback: Callable[[ModuleType], None]) -> None:
        self.m_module_name = module_name
        self.m_callback = callback
        self.m_installed = False

    def find_spec(self, fullname: str, path: Any, target: Any = None):  # type: ignore[no-untyped-def]
        if fullname != self.m_module_name:
            return None
        if fullname in sys.modules:
            self.m_callback(sys.modules[fullname])
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return spec
        if isinstance(spec.loader, _PostImportLoader):
            return spec
        spec.loader = _PostImportLoader(spec.loader, self.m_callback)
        return spec


def _install_post_import_hook(module_name: str, callback: Callable[[ModuleType], None]) -> None:
    for finder in sys.meta_path:
        if isinstance(finder, _PostImportFinder) and finder.m_module_name == module_name:
            return
    sys.meta_path.insert(0, _PostImportFinder(module_name, callback))


def _choose_max_pipeline_parallel_size(num_layers: int, preferred: int = 8) -> int:
    for candidate in [preferred, 4, 2, 1]:
        if num_layers % candidate == 0:
            return candidate
    return 1


def _patch_vidur_sequence_proxy(module: ModuleType) -> None:
    seq_proxy = getattr(module, "SequenceProxy", None)
    if seq_proxy is None:
        return
    if hasattr(seq_proxy, "get_num_prompt_tokens_stage_processed"):
        return
    if not hasattr(seq_proxy, "get_num_prompt_tokens_processed"):
        return
    setattr(
        seq_proxy,
        "get_num_prompt_tokens_stage_processed",
        getattr(seq_proxy, "get_num_prompt_tokens_processed"),
    )


def _patch_sarathi_attention_module(module: ModuleType) -> None:
    flashinfer_cls = getattr(module, "FlashinferAttentionWrapper", None)
    no_op_cls = getattr(module, "NoOpAttentionWrapper", None)
    attention_backend = getattr(module, "AttentionBackend", None)

    if flashinfer_cls is None or no_op_cls is None or attention_backend is None:
        return

    # Only patch when Sarathi's module is missing the expected singleton API (broken state).
    if hasattr(flashinfer_cls, "get_instance") and hasattr(no_op_cls, "get_instance"):
        return

    class _VidurModelConfigAdapter:
        """Adapter for Vidur's lightweight ModelConfig to Sarathi's wrapper expectations."""

        def __init__(self, model_config: Any, *, max_pipeline_parallel_size: int) -> None:
            self.m_model_config = model_config
            self.m_max_pipeline_parallel_size = int(max_pipeline_parallel_size)

        def get_num_q_heads(self, parallel_config: Any) -> int:
            return int(self.m_model_config.get_num_q_heads(parallel_config))

        def get_num_kv_heads(self, parallel_config: Any) -> int:
            return int(self.m_model_config.get_num_kv_heads(parallel_config))

        def get_head_size(self) -> int:
            return int(self.m_model_config.get_head_size())

        def get_num_layers(self, parallel_config: Any) -> int:  # noqa: ARG002
            total_layers = int(getattr(self.m_model_config, "num_layers"))
            return max(1, total_layers // self.m_max_pipeline_parallel_size)

        @property
        def dtype(self):  # type: ignore[no-untyped-def]
            return self.m_model_config.dtype

    class _VidurAttentionWrapperCompat:
        """Sarathi attention wrapper singleton compatible with Vidur attention profiling."""

        def __init__(self) -> None:
            self.m_impl: Any | None = None
            self.m_model_config: Any | None = None
            self.m_parallel_config: Any | None = None
            self.m_block_size: int | None = None
            self.m_device: Any | None = None
            self.m_layer_cache_idx: int = 0

        def init(self, model_config: Any, parallel_config: Any, block_size: int, device: Any) -> None:
            self.m_model_config = model_config
            self.m_parallel_config = parallel_config
            self.m_block_size = int(block_size)
            self.m_device = device
            self.m_impl = None

        def get_cache_block(self, num_blocks: int, **_: Any) -> int:
            if self.m_model_config is None or self.m_parallel_config is None or self.m_block_size is None:
                raise RuntimeError("Sarathi attention wrapper was used before init().")

            max_pp = _choose_max_pipeline_parallel_size(int(getattr(self.m_model_config, "num_layers", 1)))
            model_cfg = _VidurModelConfigAdapter(self.m_model_config, max_pipeline_parallel_size=max_pp)

            from sarathi.config import CacheConfig
            from sarathi.model_executor.attention.flashinfer_attention_wrapper import (
                FlashinferAttentionWrapper as _FlashinferImpl,
            )

            cache_cfg = CacheConfig(block_size=self.m_block_size, num_gpu_blocks=int(num_blocks))
            self.m_impl = _FlashinferImpl(model_cfg, self.m_parallel_config, cache_cfg, self.m_device)
            self.m_impl.init_gpu_cache(int(num_blocks))
            self.m_layer_cache_idx = 0
            return self.m_layer_cache_idx

        def begin_forward(self, seq_metadata_list: Any) -> None:
            if self.m_impl is None:
                raise RuntimeError("Sarathi attention wrapper was used before get_cache_block().")
            self.m_impl.begin_forward(seq_metadata_list)

        def end_forward(self) -> None:
            if self.m_impl is None:
                raise RuntimeError("Sarathi attention wrapper was used before get_cache_block().")
            self.m_impl.end_forward()

        def forward(self, query: Any, key: Any, value: Any, kv_cache: Any):  # type: ignore[no-untyped-def]
            if self.m_impl is None:
                raise RuntimeError("Sarathi attention wrapper was used before get_cache_block().")
            layer_idx = self.m_layer_cache_idx
            if isinstance(kv_cache, int):
                layer_idx = int(kv_cache)
            return self.m_impl.forward(query, key, value, layer_idx)

    class _VidurNoOpAttentionWrapperCompat:
        """No-op attention wrapper singleton compatible with Vidur attention profiling."""

        def __init__(self) -> None:
            self.m_device: Any | None = None

        def init(self, model_config: Any, parallel_config: Any, block_size: int, device: Any) -> None:  # noqa: ARG002
            self.m_device = device

        def get_cache_block(self, num_blocks: int, **_: Any) -> int:  # noqa: ARG002
            return 0

        def begin_forward(self, seq_metadata_list: Any) -> None:  # noqa: ARG002
            return

        def end_forward(self) -> None:
            return

        def forward(self, query: Any, key: Any, value: Any, kv_cache: Any):  # noqa: ARG002  # type: ignore[no-untyped-def]
            import torch

            device = self.m_device if self.m_device is not None else torch.device("cuda")
            return torch.empty_like(query, device=device)

    flashinfer_singleton: _VidurAttentionWrapperCompat | None = None
    no_op_singleton: _VidurNoOpAttentionWrapperCompat | None = None

    def _flashinfer_get_instance(cls):  # type: ignore[no-untyped-def]  # noqa: ARG001
        nonlocal flashinfer_singleton
        if flashinfer_singleton is None:
            flashinfer_singleton = _VidurAttentionWrapperCompat()
        return flashinfer_singleton

    def _no_op_get_instance(cls):  # type: ignore[no-untyped-def]  # noqa: ARG001
        nonlocal no_op_singleton
        if no_op_singleton is None:
            no_op_singleton = _VidurNoOpAttentionWrapperCompat()
        return no_op_singleton

    setattr(flashinfer_cls, "get_instance", classmethod(_flashinfer_get_instance))
    setattr(no_op_cls, "get_instance", classmethod(_no_op_get_instance))


_install_post_import_hook("vidur.profiling.attention.sequence_proxy", _patch_vidur_sequence_proxy)
_install_post_import_hook("sarathi.model_executor.attention", _patch_sarathi_attention_module)

