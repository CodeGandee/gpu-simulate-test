from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_scale_group_overrides_trace_subset() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "configs" / "paper_fidelity"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        small = compose(config_name="repro", overrides=["scale=small"])
        assert small.scale == "small"
        assert small.trace_subset.kind == "range"
        assert int(small.trace_subset.begin) == 0
        assert int(small.trace_subset.end) == 50

        medium = compose(config_name="repro", overrides=["scale=medium"])
        assert medium.scale == "medium"
        assert medium.trace_subset.kind == "range"
        assert int(medium.trace_subset.begin) == 0
        assert int(medium.trace_subset.end) == 500

        full = compose(config_name="repro", overrides=["scale=full"])
        assert full.scale == "full"
        assert full.trace_subset.kind == "all"

