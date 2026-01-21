from __future__ import annotations

from omegaconf import OmegaConf

from gpu_simulate_test.vidur_cli.errors import UserFacingError
from gpu_simulate_test.vidur_cli.stages import _resolve_profiling_settings


def test_resolve_profiling_settings_uses_config_when_no_cli_override() -> None:
    cfg = OmegaConf.create(
        {
            "profiling": {
                "num_gpus": 1,
                "tensor_parallel_size": 1,
                "max_tokens": 4096,
                "include_network": True,
                "cpu_overhead": {"enabled": False, "max_batch_size": 64, "validation": "warn"},
                "attention": {
                    "profile_mode": "both",
                    "backend": "FLASHINFER",
                    "block_size": 16,
                    "min_batch_size": 1,
                    "max_batch_size": 4,
                },
            },
            "hardware": {"network_device": "a100_pairwise_nvlink"},
        }
    )
    resolved = _resolve_profiling_settings(cfg, include_cpu_overhead_override=None)
    assert resolved.include_cpu_overhead is False
    assert resolved.cpu_overhead_max_batch_size == 64
    assert resolved.cpu_overhead_validation == "warn"


def test_resolve_profiling_settings_cli_override_wins() -> None:
    cfg = OmegaConf.create(
        {
            "profiling": {
                "cpu_overhead": {"enabled": False},
                "attention": {},
            },
            "hardware": {"network_device": "a100_pairwise_nvlink"},
        }
    )
    resolved = _resolve_profiling_settings(cfg, include_cpu_overhead_override=True)
    assert resolved.include_cpu_overhead is True


def test_resolve_profiling_settings_requires_network_device() -> None:
    cfg = OmegaConf.create({"profiling": {"cpu_overhead": {}, "attention": {}}})
    try:
        _resolve_profiling_settings(cfg, include_cpu_overhead_override=None)
    except UserFacingError as e:
        assert "hardware.network_device" in str(e)
    else:
        raise AssertionError("expected UserFacingError")


def test_resolve_profiling_settings_rejects_invalid_attention_batch_range() -> None:
    cfg = OmegaConf.create(
        {
            "profiling": {
                "attention": {"min_batch_size": 4, "max_batch_size": 1},
                "cpu_overhead": {},
            },
            "hardware": {"network_device": "a100_pairwise_nvlink"},
        }
    )
    try:
        _resolve_profiling_settings(cfg, include_cpu_overhead_override=None)
    except UserFacingError as e:
        assert "profiling.attention.max_batch_size" in str(e)
    else:
        raise AssertionError("expected UserFacingError")
