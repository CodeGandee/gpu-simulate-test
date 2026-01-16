from __future__ import annotations

from pathlib import Path


def run_vidur_main(*, repo_root: Path) -> None:
    from vidur.main import main as vidur_main

    vidur_main()
