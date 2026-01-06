"""
Run Vidur vendor simulations for static fidelity (static arrivals + vLLM scheduler).

This script is a "paper-style" sanity check for the static fidelity plots:

- Metric: request_execution_plus_preemption_time_normalized
- Percentiles: P50 / P95

It runs Vidur via `python -m vidur.main` using the profiling bundle shipped in the
Vidur submodule and processed trace-length CSVs under `extern/tracked/vidur/data/processed_traces`.

Outputs
-------
- Raw simulation outputs: `tmp/vendor-static-vllm/<gpu>-<model>-<trace>/...`
- Summary index: `tmp/vendor-static-vllm/manifest.json`
- Figures (SVG): `context/tasks/working/002-reproduce-vidur-paper-fidelity/figures/`

Notes
-----
- We do not run real inference; the "Real" series is plotted as 0.0 to preserve
  the paper-style figure layout.
- The Vidur submodule in this repo does not ship the paper's Chat-1M / BWB-4K
  processed traces, so only the traces present in `data/processed_traces/*.csv`
  are included.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must set backend first)


_TIMESTAMP_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}$")
_EXCLUDED_TRACE_STEMS = {
    "splitwise_code",
    "splitwise_conv",
}


@dataclass(frozen=True)
class ModelRunConfig:
    """Per-model settings for Vidur runs."""

    tensor_parallel_size: int
    num_pipeline_stages: int
    max_tokens: int
    network_device: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sanitize_component(text: str) -> str:
    text = text.replace("/", "__")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_")


def _discover_processed_traces(*, repo_root: Path) -> list[Path]:
    traces_root = repo_root / "extern" / "tracked" / "vidur" / "data" / "processed_traces"
    if not traces_root.exists():
        raise FileNotFoundError(f"Vidur processed_traces dir not found: {traces_root}")
    traces = [p for p in sorted(traces_root.glob("*.csv")) if p.stem not in _EXCLUDED_TRACE_STEMS]
    if not traces:
        raise FileNotFoundError(
            "No processed traces found after filtering excluded trace stems. "
            f"processed_traces={traces_root} excluded={sorted(_EXCLUDED_TRACE_STEMS)}"
        )
    return traces


def _paper_model_run_configs() -> dict[str, ModelRunConfig]:
    return {
        "meta-llama/Llama-2-7b-hf": ModelRunConfig(
            tensor_parallel_size=1,
            num_pipeline_stages=1,
            max_tokens=4096,
            network_device="a100_pairwise_nvlink",
        ),
        "internlm/internlm-20b": ModelRunConfig(
            tensor_parallel_size=2,
            num_pipeline_stages=1,
            max_tokens=4096,
            network_device="a100_pairwise_nvlink",
        ),
        "meta-llama/Llama-2-70b-hf": ModelRunConfig(
            tensor_parallel_size=4,
            num_pipeline_stages=1,
            max_tokens=4096,
            network_device="a100_pairwise_nvlink",
        ),
        "Qwen/Qwen-72B": ModelRunConfig(
            tensor_parallel_size=4,
            num_pipeline_stages=1,
            max_tokens=4096,
            network_device="a100_pairwise_nvlink",
        ),
    }


def _build_vidur_cmd(
    *,
    repo_root: Path,
    device: str,
    model_name: str,
    trace_file: Path,
    out_root: Path,
    cache_dir: Path,
    num_requests: int,
    model_cfg: ModelRunConfig,
) -> list[str]:
    profiling_root = repo_root / "extern" / "tracked" / "vidur" / "data" / "profiling"

    compute_file = profiling_root / "compute/{DEVICE}/{MODEL}/mlp.csv"
    attention_file = profiling_root / "compute/{DEVICE}/{MODEL}/attention.csv"
    all_reduce_file = profiling_root / "network/{NETWORK_DEVICE}/all_reduce.csv"
    send_recv_file = profiling_root / "network/{NETWORK_DEVICE}/send_recv.csv"
    cpu_overhead_file = profiling_root / "cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv"

    max_tokens = int(model_cfg.max_tokens)

    return [
        sys.executable,
        "-m",
        "vidur.main",
        "--replica_config_device",
        device,
        "--replica_config_model_name",
        model_name,
        "--replica_config_network_device",
        model_cfg.network_device,
        "--cluster_config_num_replicas",
        "1",
        "--replica_config_tensor_parallel_size",
        str(int(model_cfg.tensor_parallel_size)),
        "--replica_config_num_pipeline_stages",
        str(int(model_cfg.num_pipeline_stages)),
        "--request_generator_config_type",
        "synthetic",
        "--synthetic_request_generator_config_num_requests",
        str(int(num_requests)),
        "--length_generator_config_type",
        "trace",
        "--trace_request_length_generator_config_max_tokens",
        str(max_tokens),
        "--trace_request_length_generator_config_trace_file",
        str(trace_file.resolve()),
        "--interval_generator_config_type",
        "static",
        "--replica_scheduler_config_type",
        "vllm",
        "--vllm_scheduler_config_batch_size_cap",
        "512",
        "--vllm_scheduler_config_max_tokens_in_batch",
        "4096",
        "--random_forrest_execution_time_predictor_config_compute_input_file",
        str(compute_file.resolve()),
        "--random_forrest_execution_time_predictor_config_attention_input_file",
        str(attention_file.resolve()),
        "--random_forrest_execution_time_predictor_config_all_reduce_input_file",
        str(all_reduce_file.resolve()),
        "--random_forrest_execution_time_predictor_config_send_recv_input_file",
        str(send_recv_file.resolve()),
        "--random_forrest_execution_time_predictor_config_cpu_overhead_input_file",
        str(cpu_overhead_file.resolve()),
        "--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size",
        str(max_tokens),
        "--random_forrest_execution_time_predictor_config_prediction_max_batch_size",
        "512",
        "--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request",
        str(max_tokens),
        "--metrics_config_output_dir",
        str(out_root.resolve()),
        "--metrics_config_cache_dir",
        str(cache_dir.resolve()),
        "--no-metrics_config_store_plots",
    ]


def _find_latest_output_dir(out_root: Path) -> Path | None:
    if not out_root.exists():
        return None

    candidates: list[Path] = []
    for child in out_root.iterdir():
        if child.is_dir() and _TIMESTAMP_DIR_RE.match(child.name):
            req_csv = child / "request_metrics.csv"
            cfg = child / "config.json"
            if req_csv.exists() and cfg.exists():
                candidates.append(child)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_model_title(model_name: str, *, tp: int) -> str:
    if model_name == "meta-llama/Llama-2-7b-hf":
        base = "LLaMA2-7B"
    elif model_name == "internlm/internlm-20b":
        base = "InternLM-20B"
    elif model_name == "meta-llama/Llama-2-70b-hf":
        base = "LLaMA2-70B"
    elif model_name == "Qwen/Qwen-72B":
        base = "Qwen-72B"
    else:
        base = model_name.split("/")[-1]
    return f"{base} (TP{tp})"


def _format_trace_label(trace_file: str) -> str:
    stem = Path(trace_file).stem
    if stem == "arxiv_summarization_stats_llama2_tokenizer_filtered_v2":
        return "Arxiv-4K"
    if stem == "splitwise_conv":
        return "Splitwise-Conv"
    if stem == "splitwise_code":
        return "Splitwise-Code"
    return stem


def _plot_static_fidelity(
    *,
    runs: list[dict[str, Any]],
    metric: str,
    percentile: float,
    out_svg: Path,
    title_suffix: str,
) -> None:
    model_order = [
        "meta-llama/Llama-2-7b-hf",
        "internlm/internlm-20b",
        "meta-llama/Llama-2-70b-hf",
        "Qwen/Qwen-72B",
    ]
    trace_order = sorted({Path(r["trace_file"]).stem for r in runs})
    trace_labels = [_format_trace_label(t) for t in trace_order]

    values: dict[tuple[str, str], float] = {}
    tp_by_model: dict[str, int] = {}
    for r in runs:
        model = str(r["model_name"])
        trace_stem = Path(str(r["trace_file"])).stem
        tp_by_model[model] = int(r["model_config"]["tensor_parallel_size"])

        df = pd.read_csv(Path(r["request_metrics_csv"]))
        if metric not in df.columns:
            raise ValueError(f"{r['request_metrics_csv']}: missing metric column: {metric}")
        values[(model, trace_stem)] = float(df[metric].astype(float).quantile(percentile))

    all_pred = [values[(m, t)] for m in model_order for t in trace_order if (m, t) in values]
    y_max = max(all_pred + [0.0]) * 1.10 if all_pred else 1.0
    if y_max == 0.0:
        y_max = 1.0

    fig, axes = plt.subplots(1, 4, figsize=(11.7, 2.6), sharey=True)
    x = np.arange(len(trace_order), dtype=float)
    bar_w = 0.35

    for ax, model in zip(axes, model_order, strict=True):
        pred = [values.get((model, t), float("nan")) for t in trace_order]
        real = [0.0 for _ in trace_order]

        ax.bar(x - bar_w / 2.0, real, width=bar_w, label="Real", color="#224ed6")
        ax.bar(x + bar_w / 2.0, pred, width=bar_w, label="Predicted", color="#df7d20")

        title = _format_model_title(model, tp=tp_by_model.get(model, 1))
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x, trace_labels, fontsize=8)
        ax.grid(axis="y", color="0.85", linewidth=0.8)
        ax.set_ylim(0.0, y_max)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Normalized latency (s/token)", fontsize=9)
    axes[0].legend(loc="upper left", fontsize=8, frameon=True)

    fig.suptitle(f"Vendor (static + vLLM) {title_suffix}", fontsize=10, y=1.02)
    fig.tight_layout()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vidur vendor static fidelity (static arrivals + vLLM).")
    parser.add_argument("--device", default="a100", help="Vidur device SKU name (default: a100).")
    parser.add_argument("--num-requests", type=int, default=512, help="Synthetic request count per run.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip running if a prior output directory already exists under the run root.",
    )
    parser.add_argument(
        "--out-root",
        default=str(_repo_root() / "tmp" / "vendor-static-vllm"),
        help="Output root directory (default: tmp/vendor-static-vllm).",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    device = str(args.device)
    num_requests = int(args.num_requests)
    out_root = Path(args.out_root).expanduser().resolve()

    out_root.mkdir(parents=True, exist_ok=True)
    cache_root = out_root / "vidur-cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    traces = _discover_processed_traces(repo_root=repo_root)
    model_cfg_map = _paper_model_run_configs()

    summary: dict[str, Any] = {
        "device": device,
        "num_requests": num_requests,
        "repo_root": str(repo_root),
        "runs": [],
    }

    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"

    for model_name, model_cfg in model_cfg_map.items():
        cache_dir = cache_root / f"{device}-{_sanitize_component(model_name)}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        for trace_file in traces:
            run_root = out_root / f"{device}-{_sanitize_component(model_name)}-{trace_file.stem}"

            if args.skip_existing:
                existing = _find_latest_output_dir(run_root)
                if existing is not None:
                    summary["runs"].append(
                        {
                            "ok": True,
                            "skipped": True,
                            "device": device,
                            "model_name": model_name,
                            "model_config": {
                                "tensor_parallel_size": model_cfg.tensor_parallel_size,
                                "num_pipeline_stages": model_cfg.num_pipeline_stages,
                                "max_tokens": model_cfg.max_tokens,
                                "network_device": model_cfg.network_device,
                            },
                            "trace_file": str(trace_file.resolve()),
                            "out_root": str(run_root.resolve()),
                            "output_dir": str(existing.resolve()),
                            "request_metrics_csv": str((existing / "request_metrics.csv").resolve()),
                        }
                    )
                    continue

            run_root.mkdir(parents=True, exist_ok=True)
            cmd = _build_vidur_cmd(
                repo_root=repo_root,
                device=device,
                model_name=model_name,
                trace_file=trace_file,
                out_root=run_root,
                cache_dir=cache_dir,
                num_requests=num_requests,
                model_cfg=model_cfg,
            )

            result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            ok = result.returncode == 0

            log_path = run_root / "run.log"
            log_path.write_text(result.stdout, encoding="utf-8")

            output_dir = _find_latest_output_dir(run_root)
            run_record: dict[str, Any] = {
                "ok": ok,
                "returncode": int(result.returncode),
                "device": device,
                "model_name": model_name,
                "model_config": {
                    "tensor_parallel_size": model_cfg.tensor_parallel_size,
                    "num_pipeline_stages": model_cfg.num_pipeline_stages,
                    "max_tokens": model_cfg.max_tokens,
                    "network_device": model_cfg.network_device,
                },
                "trace_file": str(trace_file.resolve()),
                "out_root": str(run_root.resolve()),
                "cache_dir": str(cache_dir.resolve()),
                "cmd": cmd,
                "output_dir": str(output_dir.resolve()) if output_dir else None,
                "request_metrics_csv": str((output_dir / "request_metrics.csv").resolve()) if output_dir else None,
            }
            summary["runs"].append(run_record)

            if not ok:
                raise RuntimeError(f"Vidur run failed for {model_name} + {trace_file} (see {log_path})")

    _write_json(out_root / "manifest.json", summary)

    figures_dir = (
        repo_root
        / "context"
        / "tasks"
        / "working"
        / "002-reproduce-vidur-paper-fidelity"
        / "figures"
    )

    ok_runs = [r for r in summary["runs"] if r.get("ok") and r.get("request_metrics_csv")]
    metric = "request_execution_plus_preemption_time_normalized"

    _plot_static_fidelity(
        runs=ok_runs,
        metric=metric,
        percentile=0.5,
        out_svg=figures_dir / "vendor_a100_vllm_static_fidelity_request_execution_plus_preemption_time_normalized_p50.svg",
        title_suffix="request_execution_plus_preemption_time_normalized (P50)",
    )
    _plot_static_fidelity(
        runs=ok_runs,
        metric=metric,
        percentile=0.95,
        out_svg=figures_dir / "vendor_a100_vllm_static_fidelity_request_execution_plus_preemption_time_normalized_p95.svg",
        title_suffix="request_execution_plus_preemption_time_normalized (P95)",
    )


if __name__ == "__main__":
    main()
