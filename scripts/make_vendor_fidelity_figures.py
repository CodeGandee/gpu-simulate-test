"""
Generate Vidur paper-style fidelity figures from vendor simulation outputs.

This script reads Vidur simulator outputs produced via `python -m vidur.main`
and stored under `results/raw/vendor-results/sarathi-serve/dynamic/**`, aggregates
P50/P95 metrics, and renders paper-style multi-subplot bar charts as SVGs.

Notes
-----
- These plots are intended for "sim-only" sanity checking and visualization.
- If real (Sarathi) metrics are unavailable, we still plot a "Real" series as 0.0
  so the chart layout matches the paper figures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must set backend first)

_EXCLUDED_TRACE_STEMS = {
    "splitwise_code",
    "splitwise_conv",
}


@dataclass(frozen=True)
class VendorRun:
    """A single vendor Vidur simulation output directory."""

    device: str
    model_name: str
    tensor_parallel_size: int
    trace_file: str
    interval_generator: str
    qps: float | None
    run_dir: Path

    p50: dict[str, float]
    p95: dict[str, float]


def _format_model_title(model_name: str, *, tp: int) -> str:
    name = model_name
    if "Llama-2-7b" in model_name:
        name = "LLaMA2-7B"
    elif "Llama-2-70b" in model_name:
        name = "LLaMA2-70B"
    elif "Meta-Llama-3-8B" in model_name:
        name = "LLaMA3-8B"
    elif "Meta-Llama-3-70B" in model_name:
        name = "LLaMA3-70B"
    return f"{name} (TP{tp})"


def _format_trace_label(trace_file: str) -> str:
    stem = Path(trace_file).stem
    if stem == "arxiv_summarization_stats_llama2_tokenizer_filtered_v2":
        return "Arxiv-4K"
    if stem == "splitwise_conv":
        return "Splitwise-Conv"
    if stem == "splitwise_code":
        return "Splitwise-Code"
    return stem


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_run_dir(root: Path) -> Path:
    candidates = sorted([p for p in root.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"no run subdirectories found under {root}")
    return candidates[-1]


def load_vendor_runs(vendor_root: Path) -> list[VendorRun]:
    """Discover and load vendor runs under `vendor_root`."""
    runs: list[VendorRun] = []
    for entry in sorted(vendor_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in {"cache", "__pycache__"}:
            continue

        run_dir = _find_latest_run_dir(entry)
        cfg = _load_json(run_dir / "config.json")
        metrics_csv = run_dir / "request_metrics.csv"
        if not metrics_csv.exists():
            raise FileNotFoundError(f"missing request_metrics.csv: {metrics_csv}")

        replica_cfg = cfg["cluster_config"]["replica_config"]
        request_gen_cfg = cfg["request_generator_config"]
        length_cfg = request_gen_cfg["length_generator_config"]
        interval_cfg = request_gen_cfg["interval_generator_config"]

        trace_file = str(length_cfg["trace_file"])
        if Path(trace_file).stem in _EXCLUDED_TRACE_STEMS:
            continue

        df = pd.read_csv(metrics_csv)
        needed = [
            "request_execution_plus_preemption_time_normalized",
            "request_e2e_time_normalized",
        ]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{metrics_csv}: missing required columns: {missing}")

        p50 = {m: float(df[m].astype(float).quantile(0.5)) for m in needed}
        p95 = {m: float(df[m].astype(float).quantile(0.95)) for m in needed}

        runs.append(
            VendorRun(
                device=str(replica_cfg["device"]),
                model_name=str(replica_cfg["model_name"]),
                tensor_parallel_size=int(replica_cfg["tensor_parallel_size"]),
                trace_file=trace_file,
                interval_generator=str(interval_cfg["name"]),
                qps=float(interval_cfg["qps"]) if "qps" in interval_cfg else None,
                run_dir=run_dir,
                p50=p50,
                p95=p95,
            )
        )

    if not runs:
        raise FileNotFoundError(f"no vendor runs found under {vendor_root}")
    return runs


def _plot_paper_style_multi_subplot_bars(
    *,
    runs: list[VendorRun],
    device: str,
    metric: str,
    percentile: str,
    out_svg: Path,
    title_prefix: str,
) -> None:
    series_real_label = "Real"
    series_pred_label = "Predicted"

    runs = [r for r in runs if r.device == device]
    if not runs:
        raise ValueError(f"no runs found for device={device!r}")

    model_order = sorted(
        {r.model_name for r in runs},
        key=lambda s: (("70b" not in s.lower()), s.lower()),
    )
    if len(model_order) != 4:
        raise ValueError(f"expected 4 models for plotting, found {len(model_order)}: {model_order}")

    trace_order = sorted({Path(r.trace_file).stem for r in runs})
    trace_labels = [_format_trace_label(t) for t in trace_order]

    values_by_model_trace: dict[tuple[str, str], float] = {}
    for r in runs:
        key = (r.model_name, Path(r.trace_file).stem)
        if percentile == "p50":
            values_by_model_trace[key] = float(r.p50[metric])
        elif percentile == "p95":
            values_by_model_trace[key] = float(r.p95[metric])
        else:
            raise ValueError(f"unsupported percentile: {percentile}")

    all_pred = [values_by_model_trace[(m, t)] for m in model_order for t in trace_order]
    y_max = max(all_pred + [0.0]) * 1.10 if all_pred else 1.0
    if y_max == 0.0:
        y_max = 1.0

    fig, axes = plt.subplots(1, 4, figsize=(11.7, 2.6), sharey=True)
    x = np.arange(len(trace_order), dtype=float)
    bar_w = 0.35

    for ax, model_name in zip(axes, model_order, strict=True):
        pred = [values_by_model_trace[(model_name, t)] for t in trace_order]
        real = [0.0 for _ in trace_order]

        ax.bar(x - bar_w / 2.0, real, width=bar_w, label=series_real_label, color="#224ed6")
        ax.bar(x + bar_w / 2.0, pred, width=bar_w, label=series_pred_label, color="#df7d20")

        title = _format_model_title(model_name, tp=next(r.tensor_parallel_size for r in runs if r.model_name == model_name))
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x, trace_labels, fontsize=8)
        ax.grid(axis="y", color="0.85", linewidth=0.8)
        ax.set_ylim(0.0, y_max)
        ax.tick_params(axis="y", labelsize=8)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Normalized latency (s/token)", fontsize=9)
    axes[0].legend(loc="upper left", fontsize=8, frameon=True)

    fig.suptitle(f"{title_prefix} ({percentile.upper()})", fontsize=10, y=1.02)
    fig.tight_layout()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    vendor_root = repo_root / "results" / "raw" / "vendor-results" / "sarathi-serve" / "dynamic"
    out_dir = repo_root / "context" / "tasks" / "working" / "002-reproduce-vidur-paper-fidelity" / "figures"

    runs = load_vendor_runs(vendor_root)

    _plot_paper_style_multi_subplot_bars(
        runs=runs,
        device="a100",
        metric="request_execution_plus_preemption_time_normalized",
        percentile="p50",
        out_svg=out_dir
        / "vendor_a100_static_fidelity_request_execution_plus_preemption_time_normalized_p50.svg",
        title_prefix="Vendor (sim-only) static fidelity",
    )
    _plot_paper_style_multi_subplot_bars(
        runs=runs,
        device="a100",
        metric="request_execution_plus_preemption_time_normalized",
        percentile="p95",
        out_svg=out_dir
        / "vendor_a100_static_fidelity_request_execution_plus_preemption_time_normalized_p95.svg",
        title_prefix="Vendor (sim-only) static fidelity",
    )
    _plot_paper_style_multi_subplot_bars(
        runs=runs,
        device="a100",
        metric="request_e2e_time_normalized",
        percentile="p50",
        out_svg=out_dir / "vendor_a100_dynamic_fidelity_request_e2e_time_normalized_p50.svg",
        title_prefix="Vendor (sim-only) dynamic fidelity",
    )
    _plot_paper_style_multi_subplot_bars(
        runs=runs,
        device="a100",
        metric="request_e2e_time_normalized",
        percentile="p95",
        out_svg=out_dir / "vendor_a100_dynamic_fidelity_request_e2e_time_normalized_p95.svg",
        title_prefix="Vendor (sim-only) dynamic fidelity",
    )


if __name__ == "__main__":
    main()
