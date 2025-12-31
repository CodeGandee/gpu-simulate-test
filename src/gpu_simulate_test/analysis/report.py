from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gpu_simulate_test.analysis.compare import compute_percentiles
from gpu_simulate_test.analysis.plots import plot_cdf


@dataclass(frozen=True)
class ReportPaths:
    out_dir: Path
    summary_md: Path
    tables_dir: Path
    figs_dir: Path


def _percentile_table(*, real: pd.Series, sim: pd.Series, percentiles: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [f"p{int(p * 100)}" for p in percentiles],
            "real_ns": compute_percentiles(real, percentiles).to_list(),
            "sim_ns": compute_percentiles(sim, percentiles).to_list(),
        }
    )


def write_summary_md(
    paths: ReportPaths,
    *,
    ttft_table: pd.DataFrame,
    token_table: pd.DataFrame,
    real_run_dir: Path,
    sim_run_dir: Path,
) -> None:
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    paths.figs_dir.mkdir(parents=True, exist_ok=True)

    def _md_table(df: pd.DataFrame) -> str:
        header = "| " + " | ".join(df.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        rows = ["| " + " | ".join(str(x) for x in row) + " |" for row in df.to_numpy().tolist()]
        return "\n".join([header, sep, *rows])

    content = [
        "# Compare runs summary",
        "",
        f"- real: `{real_run_dir}`",
        f"- sim: `{sim_run_dir}`",
        "",
        "## TTFT percentiles (ns)",
        "",
        _md_table(ttft_table),
        "",
        "## Decode token latency percentiles (ns)",
        "",
        _md_table(token_table),
        "",
        "Notes:",
        "- Token alignment truncates sim tokens using real `num_decode_tokens_actual` per request.",
        "",
    ]
    paths.summary_md.write_text("\n".join(content), encoding="utf-8")


def write_report(
    paths: ReportPaths,
    *,
    ttft_real: pd.Series,
    ttft_sim: pd.Series,
    token_lat_real: pd.Series,
    token_lat_sim: pd.Series,
    percentiles: list[float],
    real_run_dir: Path,
    sim_run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ttft_table = _percentile_table(real=ttft_real, sim=ttft_sim, percentiles=percentiles)
    token_table = _percentile_table(real=token_lat_real, sim=token_lat_sim, percentiles=percentiles)

    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    ttft_table.to_csv(paths.tables_dir / "ttft_percentiles.csv", index=False)
    token_table.to_csv(paths.tables_dir / "token_latency_percentiles.csv", index=False)

    plot_cdf(
        real=ttft_real,
        sim=ttft_sim,
        out_path=paths.figs_dir / "ttft_cdf.png",
        title="TTFT CDF",
        xlabel="ttft_ns",
    )
    plot_cdf(
        real=token_lat_real,
        sim=token_lat_sim,
        out_path=paths.figs_dir / "token_latency_cdf.png",
        title="Decode token latency CDF",
        xlabel="token_latency_ns",
    )

    write_summary_md(
        paths,
        ttft_table=ttft_table,
        token_table=token_table,
        real_run_dir=real_run_dir,
        sim_run_dir=sim_run_dir,
    )
    return ttft_table, token_table

