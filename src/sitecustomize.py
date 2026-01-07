from __future__ import annotations

import os


def _enabled() -> bool:
    value = os.environ.get("GPU_SIMULATE_TEST_ENABLE_VIDUR_ATTENTION_COMPAT", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _patch_vidur_attention_backend() -> None:
    from sarathi.config import CacheConfig
    from sarathi.types import AttentionBackend

    import sarathi.model_executor.attention as sarathi_attention
    from sarathi.model_executor.attention.flashinfer_attention_wrapper import (
        FlashinferAttentionWrapper,
    )

    from sarathi.model_executor.attention import base_attention_wrapper as sarathi_base_attention
    from sarathi.model_executor.layers import layernorm as sarathi_layernorm
    from sarathi.model_executor.parallel_utils.tensor_parallel import layers as sarathi_tp_layers

    from vidur.profiling.common.cuda_timer import CudaTimer as VidurCudaTimer
    from vidur.profiling.attention.sequence_proxy import SequenceProxy
    from vidur.profiling.common.model_config import ModelConfig as VidurProfilingModelConfig

    if not hasattr(SequenceProxy, "get_num_prompt_tokens_stage_processed"):
        SequenceProxy.get_num_prompt_tokens_stage_processed = (  # type: ignore[assignment]
            SequenceProxy.get_num_prompt_tokens_processed
        )

    if not hasattr(VidurProfilingModelConfig, "get_num_layers"):
        VidurProfilingModelConfig.get_num_layers = (  # type: ignore[assignment]
            lambda self, parallel_config: int(self.num_layers // parallel_config.pipeline_parallel_size)
        )

    # Sarathi's timers assume a MetricsStore instance exists. Vidur's profiling timers are standalone.
    sarathi_base_attention.CudaTimer = VidurCudaTimer  # type: ignore[assignment]
    sarathi_layernorm.CudaTimer = VidurCudaTimer  # type: ignore[assignment]
    sarathi_tp_layers.CudaTimer = VidurCudaTimer  # type: ignore[assignment]

    class _VidurAttentionWrapperShim:
        def __init__(self) -> None:
            self._impl: object | None = None

        def init(self, model_config, parallel_config, block_size, device):  # type: ignore[no-untyped-def]
            backend = sarathi_attention.ATTENTION_BACKEND
            if backend != AttentionBackend.FLASHINFER:
                raise ValueError(
                    f"Unsupported attention backend for Vidur profiling on this host: {backend}"
                )
            cache_config = CacheConfig(block_size=int(block_size), num_gpu_blocks=None)
            self._impl = FlashinferAttentionWrapper(
                model_config, parallel_config, cache_config, device
            )

        def get_cache_block(self, num_blocks: int, **kwargs):  # type: ignore[no-untyped-def]
            if self._impl is None:
                raise RuntimeError("Attention wrapper not initialized; call init() first.")
            return self._impl.get_cache_block(num_blocks, **kwargs)

        def begin_forward(self, seq_metadata_list):  # type: ignore[no-untyped-def]
            if self._impl is None:
                raise RuntimeError("Attention wrapper not initialized; call init() first.")
            return self._impl.begin_forward(seq_metadata_list)

        def end_forward(self):  # type: ignore[no-untyped-def]
            if self._impl is None:
                raise RuntimeError("Attention wrapper not initialized; call init() first.")
            return self._impl.end_forward()

        def forward(self, query, key, value, kv_cache):  # type: ignore[no-untyped-def]
            if self._impl is None:
                raise RuntimeError("Attention wrapper not initialized; call init() first.")
            self._impl.gpu_cache = [kv_cache]
            return self._impl.forward(query, key, value, 0)

    shim = _VidurAttentionWrapperShim()

    def _get_attention_wrapper():  # type: ignore[no-untyped-def]
        return shim

    sarathi_attention.get_attention_wrapper = _get_attention_wrapper  # type: ignore[assignment]


if _enabled():
    try:
        _patch_vidur_attention_backend()
    except Exception as e:
        raise RuntimeError(
            "Failed to apply Vidur/Sarathi attention profiling compatibility patch. "
            "Unset GPU_SIMULATE_TEST_ENABLE_VIDUR_ATTENTION_COMPAT to disable, or fix the underlying error."
        ) from e
