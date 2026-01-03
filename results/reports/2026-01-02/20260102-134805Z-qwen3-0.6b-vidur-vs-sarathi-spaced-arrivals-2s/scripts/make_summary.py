from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Callable


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:.3f}"
    return str(value)


def _md_table(
    rows: list[dict[str, Any]],
    *,
    columns: list[str],
    max_rows: int | None = None,
    formatters: dict[str, Callable[[Any], Any]] | None = None,
) -> str:
    formatters = formatters or {}
    shown = rows if max_rows is None else rows[:max_rows]

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
    for row in shown:
        cells: list[str] = []
        for col in columns:
            val = row.get(col)
            val = formatters[col](val) if col in formatters else val
            cells.append(_format_cell(val))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    experiment_dir = Path(__file__).resolve().parents[1]

    workload_dir = experiment_dir / "workload"
    real_run_dir = experiment_dir / "real_run"
    vidur_run_dir = experiment_dir / "vidur_run"
    compare_dir = experiment_dir / "compare"

    workload_trace = _read_csv(workload_dir / "trace_intervals.csv")
    workload_lengths = _read_csv(workload_dir / "trace_lengths.csv")

    real_req = _read_csv(real_run_dir / "request_metrics.csv")
    real_tok = _read_csv(real_run_dir / "token_metrics.csv")
    real_meta = _load_json(real_run_dir / "run_meta.json")

    sim_req = _read_csv(vidur_run_dir / "request_metrics.csv")
    sim_tok = _read_csv(vidur_run_dir / "token_metrics.csv")
    sim_meta = _load_json(vidur_run_dir / "run_meta.json")

    ttft_pct = _read_csv(compare_dir / "tables" / "ttft_percentiles.csv")
    tok_pct = _read_csv(compare_dir / "tables" / "token_latency_percentiles.csv")

    def _to_int(x: Any) -> int:
        return int(float(x))

    def _ns_to_ms(x: Any) -> float:
        return float(x) / 1e6

    # Sanity: ensure the "no overlap" intent is true for the real run.
    arrival_by_req = {int(r["request_id"]): _to_int(r["arrival_time_ns"]) for r in real_req}
    completion_by_req = {int(r["request_id"]): _to_int(r["completion_time_ns"]) for r in real_req}
    non_overlap = True
    for rid in sorted(arrival_by_req.keys()):
        next_rid = rid + 1
        if next_rid not in arrival_by_req:
            continue
        if completion_by_req[rid] > arrival_by_req[next_rid]:
            non_overlap = False
            break

    vidur_submodule = experiment_dir / ".." / ".." / ".." / "extern" / "tracked" / "vidur"
    vidur_git_status = subprocess.run(
        ["git", "-C", str(vidur_submodule.resolve()), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    vidur_submodule_clean = vidur_git_status.returncode == 0 and vidur_git_status.stdout.strip() == ""

    # Keep the token excerpt small: request 0 first 8 tokens.
    def _filter_tokens(rows: list[dict[str, str]], *, request_id: int, max_tokens: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in rows:
            if int(r.get("request_id", -1)) != request_id:
                continue
            if int(r.get("token_index", 0)) >= max_tokens:
                continue
            out.append(dict(r))
        out.sort(key=lambda d: int(d["token_index"]))
        return out

    real_tok_0 = _filter_tokens(real_tok, request_id=0, max_tokens=8)
    sim_tok_0 = _filter_tokens(sim_tok, request_id=0, max_tokens=8)

    # Convert percentile rows to numeric.
    ttft_rows: list[dict[str, Any]] = []
    for r in ttft_pct:
        ttft_rows.append(
            {
                "metric": r["metric"],
                "real_ns": float(r["real_ns"]),
                "sim_ns": float(r["sim_ns"]),
                "real_ms": _ns_to_ms(float(r["real_ns"])),
                "sim_ms": _ns_to_ms(float(r["sim_ns"])),
            }
        )
    tok_rows: list[dict[str, Any]] = []
    for r in tok_pct:
        tok_rows.append(
            {
                "metric": r["metric"],
                "real_ns": float(r["real_ns"]),
                "sim_ns": float(r["sim_ns"]),
                "real_ms": _ns_to_ms(float(r["real_ns"])),
                "sim_ms": _ns_to_ms(float(r["sim_ns"])),
            }
        )

    content: list[str] = []
    content.append("# Qwen3-0.6B: Vidur sim vs Sarathi real timing (A100)")
    content.append("")
    content.append("## What this is")
    content.append("")
    content.append("- Goal: run Vidur (simulation) and Sarathi-Serve (real inference) on the same workload so timings are meaningfully comparable.")
    content.append("- Key trick: the workload uses **fixed 2s inter-arrival** so `real-bench` (which runs requests sequentially) does not introduce queueing into TTFT.")
    content.append("")
    content.append("## Run directories")
    content.append("")
    content.append(f"- workload: `{workload_dir}`")
    content.append(f"- real (sarathi): `{real_run_dir}`")
    content.append(f"- sim (vidur): `{vidur_run_dir}`")
    content.append(f"- compare: `{compare_dir}` (`compare/summary.md` has percentile tables + plots)")
    content.append("")
    content.append("## GPU vs CPU (what actually ran where)")
    content.append("")
    content.append(f"- `real_run/` is **actual GPU inference** (device: `{real_meta['env'].get('cuda_device_name')}`), measuring wall-clock after each `LLMEngine.step()`.")
    content.append("- `vidur_run/` is **CPU-side simulation** using an A100 profiling bundle + learned per-op timing models; it does **not** run real end-to-end GPU inference.")
    content.append("")
    content.append("## Workload snapshot")
    content.append("")
    content.append("`trace_intervals.csv` (arrivals):")
    content.append("")
    content.append(_md_table(workload_trace, columns=["request_id", "inter_arrival_ns", "arrival_time_ns"]))
    content.append("")
    content.append("`trace_lengths.csv` (token counts):")
    content.append("")
    content.append(_md_table(workload_lengths, columns=["request_id", "prompt_id", "num_prefill_tokens", "num_decode_tokens"]))
    content.append("")
    content.append("No-overlap check (real run):")
    content.append("")
    content.append(f"- `completion_time_ns[i] <= arrival_time_ns[i+1]` for all i: `{non_overlap}`")
    content.append("")
    content.append("Submodule diff check (Vidur):")
    content.append("")
    content.append(f"- `git -C extern/tracked/vidur status --porcelain` empty: `{vidur_submodule_clean}`")
    content.append("")
    content.append("## Key results (from `compare/tables/*.csv`)")
    content.append("")
    content.append("TTFT percentiles:")
    content.append("")
    content.append(_md_table(ttft_rows, columns=["metric", "real_ns", "sim_ns", "real_ms", "sim_ms"]))
    content.append("")
    content.append("Decode token latency percentiles:")
    content.append("")
    content.append(_md_table(tok_rows, columns=["metric", "real_ns", "sim_ns", "real_ms", "sim_ms"]))
    content.append("")
    content.append("## How to interpret the run CSVs")
    content.append("")
    content.append("### `real_run/request_metrics.csv`")
    content.append("")
    content.append(_md_table(real_req, columns=list(real_req[0].keys())))
    content.append("")
    content.append("### `vidur_run/request_metrics.csv`")
    content.append("")
    content.append(_md_table(sim_req, columns=list(sim_req[0].keys())))
    content.append("")
    content.append("### `real_run/token_metrics.csv` (request 0, first 8 tokens)")
    content.append("")
    content.append(_md_table(real_tok_0, columns=list(real_tok_0[0].keys())))
    content.append("")
    content.append("### `vidur_run/token_metrics.csv` (request 0, first 8 tokens)")
    content.append("")
    content.append(_md_table(sim_tok_0, columns=list(sim_tok_0[0].keys())))
    content.append("")
    content.append("Note: Vidur only provides request-level timing in its raw metrics; this repo’s wrapper synthesizes per-token times by linear interpolation between `first_token_time_ns` and `completion_time_ns` (see `src/gpu_simulate_test/vidur_ext/sim_runner.py`).")
    content.append("")
    content.append("## Reproduce")
    content.append("")
    content.append(f"- Run: `bash {experiment_dir / 'scripts' / 'run.sh'}`")
    content.append(f"- Regenerate this summary: `python3 {experiment_dir / 'scripts' / 'make_summary.py'}`")
    content.append("")

    (experiment_dir / "summary.md").write_text("\n".join(content), encoding="utf-8")


if __name__ == "__main__":
    main()
