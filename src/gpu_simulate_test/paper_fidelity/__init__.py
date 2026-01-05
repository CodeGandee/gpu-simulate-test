"""
Paper-fidelity reproduction workflow for the Vidur MLSys'24 methodology.

This package provides a CLI-first, artifact-driven workflow that standardizes:

- Canonical trace generation/validation (`trace.csv`)
- Simulation and real replay metric schemas (`request_metrics.csv`)
- Capacity discovery for dynamic workloads (85% operating point)
- Scoring and report writing (`summary.md`)
"""

from __future__ import annotations

__all__ = [
    "capacity",
    "paths",
    "report",
    "scoring",
    "traces",
]

