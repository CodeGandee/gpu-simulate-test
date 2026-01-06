"""
Vidur paper figure → numeric tables extractor.

This script extracts numeric values from Vidur paper figure PDFs under:

- `extern/tracked/vidur/paper/tex/graphs/`

and writes JSON tables under:

- `context/summaries/vidur-kb/paper-results/`

Intermediate artifacts (SVG conversions) are written to `tmp/` so the repo stays
clean and reproducible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


SVG_NS = {"svg": "http://www.w3.org/2000/svg"}

RECT_PATH_RE = re.compile(
    r"\s*M\s+(?P<x0>-?\d+(?:\.\d+)?)\s+(?P<y0>-?\d+(?:\.\d+)?)\s*"
    r"L\s+(?P<x1>-?\d+(?:\.\d+)?)\s+(?P<y0b>-?\d+(?:\.\d+)?)\s*"
    r"L\s+(?P<x1b>-?\d+(?:\.\d+)?)\s+(?P<y1>-?\d+(?:\.\d+)?)\s*"
    r"L\s+(?P<x0b>-?\d+(?:\.\d+)?)\s+(?P<y1b>-?\d+(?:\.\d+)?)\s*Z\s*"
)

POINT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class Transform:
    """
    SVG affine transform for `matrix(a,b,c,d,e,f)`.

    Parameters
    ----------
    a, b, c, d, e, f : float
        SVG matrix coefficients.
    """

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    def apply(self, x: float, y: float) -> tuple[float, float]:
        """
        Apply transform to a point.

        Parameters
        ----------
        x : float
            X coordinate.
        y : float
            Y coordinate.

        Returns
        -------
        tuple[float, float]
            Transformed point (x, y).
        """

        return (self.a * x + self.c * y + self.e, self.b * x + self.d * y + self.f)


@dataclass(frozen=True)
class Rect:
    """
    Axis-aligned rectangle.

    Parameters
    ----------
    x0, y0, x1, y1 : float
        Bounds (min x/y, max x/y).
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        """Return rectangle width."""

        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Return rectangle height."""

        return self.y1 - self.y0


def compute_sha256(path: Path) -> str:
    """
    Compute SHA256 checksum of a file.

    Parameters
    ----------
    path : Path
        Input file.

    Returns
    -------
    str
        Hex digest.
    """

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> None:
    """
    Ensure a directory exists.

    Parameters
    ----------
    path : Path
        Directory path.
    """

    path.mkdir(parents=True, exist_ok=True)


def run_stdout(cmd: list[str]) -> str:
    """
    Run a command and return stdout.

    Parameters
    ----------
    cmd : list[str]
        Command args.

    Returns
    -------
    str
        Stdout.
    """

    return subprocess.check_output(cmd, text=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """
    Write JSON with stable formatting.

    Parameters
    ----------
    path : Path
        Output path.
    payload : dict[str, Any]
        JSON payload.
    """

    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_transform(transform_attr: str | None) -> Transform:
    """
    Parse `transform="matrix(a,b,c,d,e,f)"` attribute.

    Parameters
    ----------
    transform_attr : str | None
        Transform attribute value.

    Returns
    -------
    Transform
        Parsed transform, or identity if missing.
    """

    if not transform_attr:
        return Transform(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    m = re.match(r"^matrix\(([^)]+)\)$", transform_attr.strip())
    if not m:
        raise ValueError(f"Unsupported SVG transform: {transform_attr}")
    parts = [float(p.strip()) for p in m.group(1).split(",")]
    if len(parts) != 6:
        raise ValueError(f"Invalid SVG matrix: {transform_attr}")
    return Transform(*parts)


def parse_rect_path(d_attr: str) -> Rect | None:
    """
    Parse a Matplotlib rectangle path into Rect.

    Parameters
    ----------
    d_attr : str
        Path `d` attribute.

    Returns
    -------
    Rect | None
        Rectangle in local coordinates, or None if not a rectangle.
    """

    m = RECT_PATH_RE.match(d_attr.strip())
    if not m:
        return None
    x0 = float(m.group("x0"))
    x1 = float(m.group("x1"))
    y0 = float(m.group("y0"))
    y1 = float(m.group("y1"))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return Rect(x0, y0, x1, y1)


def transform_rect(rect: Rect, transform: Transform) -> Rect:
    """
    Apply transform to a rectangle, returning an axis-aligned bounding box.

    Parameters
    ----------
    rect : Rect
        Rectangle in local coordinates.
    transform : Transform
        Affine transform.

    Returns
    -------
    Rect
        Axis-aligned bounding box in final SVG coordinates.
    """

    corners = [
        transform.apply(rect.x0, rect.y0),
        transform.apply(rect.x0, rect.y1),
        transform.apply(rect.x1, rect.y0),
        transform.apply(rect.x1, rect.y1),
    ]
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]
    return Rect(min(xs), min(ys), max(xs), max(ys))


def parse_points(d_attr: str, transform: Transform) -> list[tuple[float, float]]:
    """
    Parse coordinate pairs from a path and apply a transform.

    Parameters
    ----------
    d_attr : str
        Path `d` attribute.
    transform : Transform
        Affine transform.

    Returns
    -------
    list[tuple[float, float]]
        Points in final SVG coordinates.
    """

    pts = [(float(x), float(y)) for x, y in POINT_RE.findall(d_attr)]
    return [transform.apply(x, y) for x, y in pts]


def pdf_to_svg(pdf_path: Path, svg_path: Path) -> None:
    """
    Convert PDF page 1 to SVG via Poppler `pdftocairo`.

    Parameters
    ----------
    pdf_path : Path
        Source PDF.
    svg_path : Path
        Output SVG file.
    """

    ensure_dir(svg_path.parent)
    subprocess.run(
        ["pdftocairo", "-svg", "-f", "1", "-l", "1", str(pdf_path), str(svg_path)],
        check=True,
    )


def detect_plot_areas(paths: list[ET.Element]) -> list[Rect]:
    """
    Detect plot areas via white-filled rectangles (Matplotlib axes background).

    Parameters
    ----------
    paths : list[ET.Element]
        SVG path elements.

    Returns
    -------
    list[Rect]
        Plot areas sorted left-to-right.
    """

    white = "rgb(100%, 100%, 100%)"
    candidates: list[Rect] = []
    for p in paths:
        if p.get("fill") != white:
            continue
        d_attr = p.get("d")
        if not d_attr:
            continue
        rect_local = parse_rect_path(d_attr)
        if not rect_local:
            continue
        rect = transform_rect(rect_local, parse_transform(p.get("transform")))
        if rect.width < 80 or rect.height < 80:
            continue
        candidates.append(rect)

    if not candidates:
        return []

    candidates_sorted = sorted(candidates, key=lambda r: r.width * r.height)
    plot_areas = candidates_sorted[:-1] if len(candidates_sorted) > 1 else candidates_sorted
    return sorted(plot_areas, key=lambda r: r.x0)


def extract_bar_chart(
    svg_path: Path,
    y_max: float,
    model_labels: list[str],
    metric: str,
    unit: str,
    series_by_fill: dict[str, str],
    extra_metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Digitize a multi-subplot bar chart with a linear y-axis.

    Parameters
    ----------
    svg_path : Path
        Converted SVG path.
    y_max : float
        Axis max value used in the paper figure.
    model_labels : list[str]
        Subplot labels, left-to-right.
    metric : str
        Metric identifier.
    unit : str
        Unit string.
    series_by_fill : dict[str, str]
        Mapping series label -> SVG fill color string.
    extra_metadata : dict[str, Any]
        Extra metadata to include.

    Returns
    -------
    dict[str, Any]
        JSON payload.
    """

    root = ET.fromstring(svg_path.read_text(errors="ignore"))
    paths = root.findall(".//svg:path", SVG_NS)
    plot_areas = detect_plot_areas(paths)
    if not plot_areas:
        raise RuntimeError(f"Failed to detect plot areas in {svg_path}")
    if len(plot_areas) != len(model_labels):
        raise RuntimeError(
            f"Expected {len(model_labels)} plot areas, got {len(plot_areas)} in {svg_path}"
        )

    trace_labels = ["Chat-1M", "Arxiv-4K", "BWB-4K"]
    rows: list[dict[str, Any]] = []
    for subplot_idx, area in enumerate(plot_areas):
        for series, fill_color in series_by_fill.items():
            bars: list[Rect] = []
            for p in paths:
                if p.get("fill") != fill_color:
                    continue
                d_attr = p.get("d")
                if not d_attr:
                    continue
                rect_local = parse_rect_path(d_attr)
                if not rect_local:
                    continue
                rect = transform_rect(rect_local, parse_transform(p.get("transform")))
                if rect.x0 < area.x0 - 1 or rect.x1 > area.x1 + 1:
                    continue
                if rect.y0 < area.y0 - 1 or rect.y1 > area.y1 + 1:
                    continue
                # Data bars touch the baseline (bottom of plot). This filters out legend patches.
                if abs(rect.y1 - area.y1) > 0.5:
                    continue
                if rect.width > area.width * 0.25:
                    continue
                bars.append(rect)

            bars_sorted = sorted(bars, key=lambda r: r.x0)
            if len(bars_sorted) != 3:
                raise RuntimeError(
                    f"{svg_path}: subplot={subplot_idx} series={series} expected 3 bars, got {len(bars_sorted)}"
                )

            for trace, bar in zip(trace_labels, bars_sorted, strict=True):
                value = (area.y1 - bar.y0) / (area.y1 - area.y0) * y_max
                rows.append(
                    {
                        "model": model_labels[subplot_idx],
                        "trace": trace,
                        "series": series,
                        "metric": metric,
                        "unit": unit,
                        "value": value,
                    }
                )

    return {
        "figure_type": "bar_multi_subplot",
        "metric": metric,
        "unit": unit,
        "axis": {"y_max": y_max},
        "rows": rows,
        **extra_metadata,
    }


def extract_error_trends_p95(svg_path: Path) -> dict[str, Any]:
    """
    Extract error trend lines (% error) for p95 normalized latency.

    Parameters
    ----------
    svg_path : Path
        Converted SVG path.

    Returns
    -------
    dict[str, Any]
        JSON payload.
    """

    model_labels = [
        "LLaMA2-7B (TP1)",
        "InternLM-20B (TP2)",
        "LLaMA2-70B (TP4)",
        "Qwen-72B (TP4)",
    ]
    load_fracs = [0.75, 0.80, 0.85, 0.95]
    grid_stroke = "rgb(69.018555%, 69.018555%, 69.018555%)"

    trace_by_stroke = {
        "rgb(0.784302%, 24.313354%, 100%)": "Chat-1M",
        "rgb(100%, 48.626709%, 0%)": "Arxiv-4K",
        "rgb(10.195923%, 78.822327%, 21.960449%)": "BWB-4K",
    }

    root = ET.fromstring(svg_path.read_text(errors="ignore"))
    paths = root.findall(".//svg:path", SVG_NS)

    hgrid_y: list[float] = []
    for p in paths:
        if p.get("stroke") != grid_stroke:
            continue
        d_attr = p.get("d")
        if not d_attr:
            continue
        pts = parse_points(d_attr, parse_transform(p.get("transform")))
        if len(pts) < 2:
            continue
        (x0, y0), (x1, y1) = pts[0], pts[1]
        if abs(y0 - y1) > 1e-3:
            continue
        if abs(x1 - x0) < 50:
            continue
        hgrid_y.append(y0)

    y_levels = sorted({round(y, 6) for y in hgrid_y})
    if len(y_levels) < 3:
        raise RuntimeError(f"{svg_path}: failed to infer y gridlines")
    y0, _, y10 = y_levels[:3]

    # Map y position (SVG coordinates) to percent error using 0 and -10 ticks.
    a = (-10.0 - 0.0) / (y10 - y0)
    b = -a * y0

    series_paths: list[tuple[str, Rect, list[tuple[float, float]]]] = []
    for p in paths:
        if p.get("fill") not in (None, "none"):
            continue
        stroke = p.get("stroke")
        if not stroke or stroke not in trace_by_stroke:
            continue
        d_attr = p.get("d")
        if not d_attr:
            continue
        pts = parse_points(d_attr, parse_transform(p.get("transform")))
        if len(pts) < 4:
            continue
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        series_paths.append((stroke, Rect(min(xs), min(ys), max(xs), max(ys)), pts))

    if len(series_paths) != 12:
        raise RuntimeError(f"{svg_path}: expected 12 series paths, got {len(series_paths)}")

    series_paths_sorted = sorted(series_paths, key=lambda item: item[1].x0)
    groups: list[list[tuple[str, Rect, list[tuple[float, float]]]]] = []
    for item in series_paths_sorted:
        if not groups:
            groups.append([item])
            continue
        if item[1].x0 - groups[-1][0][1].x0 > 120:
            groups.append([item])
        else:
            groups[-1].append(item)

    if len(groups) != 4 or any(len(g) != 3 for g in groups):
        raise RuntimeError(f"{svg_path}: failed to group series into 4 subplots")

    rows: list[dict[str, Any]] = []
    for subplot_idx, group in enumerate(groups):
        for stroke, _, pts in group:
            pts_sorted = sorted(pts, key=lambda xy: xy[0])[:4]
            for load_frac, (_, y) in zip(load_fracs, pts_sorted, strict=True):
                rows.append(
                    {
                        "model": model_labels[subplot_idx],
                        "trace": trace_by_stroke[stroke],
                        "load_frac_of_capacity": load_frac,
                        "metric": "percent_error_normalized_latency",
                        "unit": "percent",
                        "value": a * y + b,
                    }
                )

    return {
        "figure_type": "line_multi_subplot",
        "metric": "percent_error_normalized_latency",
        "unit": "percent",
        "axis": {"y_ticks_percent": [0.0, -5.0, -10.0]},
        "rows": rows,
    }


def extract_parallel_coord_qps(text: str) -> list[dict[str, Any]]:
    """
    Extract QPS/$ values from `parallel_coord.pdf` text.

    Parameters
    ----------
    text : str
        `pdftotext` output.

    Returns
    -------
    list[dict[str, Any]]
        Rows with scenario and QPS/$.
    """

    pattern = re.compile(r"^([A-Za-z0-9\-]+):\s*([0-9]+(?:\.[0-9]+)?)\s+QPS/\$$")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        rows.append({"scenario": m.group(1), "qps_per_dollar": float(m.group(2)), "unit": "QPS/$"})
    if not rows:
        raise RuntimeError("parallel_coord.pdf: no QPS/$ rows found")
    return rows


def extract_capacity_per_dollar_from_parallel_coord(
    parallel_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert parallel-coord scenario QPS/$ into (model_size, trace) rows.

    Parameters
    ----------
    parallel_rows : list[dict[str, Any]]
        Rows from `extract_parallel_coord_qps`.

    Returns
    -------
    list[dict[str, Any]]
        Rows for the capacity-per-dollar figure.
    """

    model_map = {
        "Llama-7b": "7B",
        "InternLM-20b": "20B",
        "Llama-70b": "70B",
        "Qwen-72B": "72B",
    }
    trace_map = {
        "Chat1M": "Chat1M",
        "Arxiv": "ArxivSum",
        "BWB": "BWB",
    }

    rows: list[dict[str, Any]] = []
    for row in parallel_rows:
        scenario = str(row["scenario"])
        parts = scenario.split("-")
        if len(parts) < 3:
            continue
        model_key = "-".join(parts[:2])
        trace_key = parts[2]
        if model_key not in model_map or trace_key not in trace_map:
            continue
        rows.append(
            {
                "model_size": model_map[model_key],
                "trace": trace_map[trace_key],
                "qps_per_dollar": float(row["qps_per_dollar"]),
                "unit": "QPS/$",
            }
        )
    if len(rows) != 12:
        raise RuntimeError(f"capacity_per_dollar: expected 12 rows, got {len(rows)}")
    return sorted(rows, key=lambda r: (r["model_size"], r["trace"]))


def extract_best_config(text: str) -> dict[str, Any]:
    """
    Extract optimizer 'Best Config' and QPS/$ from a figure PDF text.

    Parameters
    ----------
    text : str
        `pdftotext` output.

    Returns
    -------
    dict[str, Any]
        Best config fields and QPS/$.
    """

    m = re.search(
        r"Best Config:\s*(.+?)\s*QPS per Dollar:\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        re.S,
    )
    if not m:
        raise RuntimeError("best_config: missing Best Config / QPS per Dollar block")
    raw_config = m.group(1).strip()
    qps_per_dollar = float(m.group(2))

    fields: dict[str, str] = {}
    for item in raw_config.split(","):
        item_stripped = item.strip()
        if not item_stripped or ":" not in item_stripped:
            continue
        k, v = item_stripped.split(":", 1)
        fields[k.strip()] = v.strip()

    return {
        "best_config_raw": raw_config,
        "best_config": fields,
        "qps_per_dollar": qps_per_dollar,
        "unit": "QPS/$",
    }


def extract_vidur_hld(text: str) -> dict[str, Any]:
    """
    Extract sample spec/report blocks from `vidur-hld.pdf`.

    Parameters
    ----------
    text : str
        `pdftotext` output.

    Returns
    -------
    dict[str, Any]
        Structured model/simulation/report fields.
    """

    key_value = re.compile(r"^([^:]+):\s*(.+)$")
    model_spec: dict[str, str] = {}
    sim_spec: dict[str, str] = {}
    sim_report: dict[str, str] = {}

    current: dict[str, str] | None = None
    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if line_stripped == "Model Spec":
            current = model_spec
            continue
        if line_stripped == "Simulation Spec":
            current = sim_spec
            continue
        if line_stripped == "Simulation Report":
            current = sim_report
            continue
        m = key_value.match(line_stripped)
        if not m or current is None:
            continue
        current[m.group(1).strip()] = m.group(2).strip()

    return {
        "model_spec": model_spec,
        "simulation_spec": sim_spec,
        "simulation_report": sim_report,
    }


def extract_confusion_matrix_bbox(bbox_html: str) -> dict[str, Any]:
    """
    Extract a 3x3 cost-ratio confusion matrix from `pdftotext -bbox` output.

    Parameters
    ----------
    bbox_html : str
        `pdftotext -bbox` output HTML.

    Returns
    -------
    dict[str, Any]
        Matrix with row/col labels and numeric values.
    """

    word_re = re.compile(
        r"<word\s+xMin=\"(?P<x>-?\d+(?:\.\d+)?)\"\s+yMin=\"(?P<y>-?\d+(?:\.\d+)?)\"[^>]*>(?P<t>[^<]+)</word>"
    )
    words: list[tuple[float, float, str]] = []
    for m in word_re.finditer(bbox_html):
        words.append((float(m.group("x")), float(m.group("y")), m.group("t").strip()))

    col_labels = [(x, t) for x, y, t in words if t in ("ArxivSum", "BWB-4K", "Chat1M") and y > 200]
    row_labels = [(y, t) for x, y, t in words if t in ("ArxivSum", "BWB-4K", "Chat1M") and x < 40]
    if len(col_labels) != 3 or len(row_labels) != 3:
        raise RuntimeError("confusion_matrix: failed to locate axis labels")

    col_labels_sorted = [t for _, t in sorted(col_labels, key=lambda it: it[0])]
    row_labels_sorted = [t for _, t in sorted(row_labels, key=lambda it: it[0])]

    col_x = [x for x, _ in sorted(col_labels, key=lambda it: it[0])]
    row_y = [y for y, _ in sorted(row_labels, key=lambda it: it[0])]

    cell_values: list[tuple[int, int, float]] = []
    for x, y, t in words:
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", t):
            continue
        if x >= 260:
            continue
        if x < 50:
            continue
        value = float(t)
        col_idx = min(range(3), key=lambda i: abs(x - col_x[i]))
        row_idx = min(range(3), key=lambda i: abs(y - row_y[i]))
        cell_values.append((row_idx, col_idx, value))

    if len(cell_values) != 9:
        raise RuntimeError(f"confusion_matrix: expected 9 values, got {len(cell_values)}")

    matrix: list[list[float | None]] = [[None, None, None] for _ in range(3)]
    for r, c, v in cell_values:
        matrix[r][c] = v
    if any(v is None for row in matrix for v in row):
        raise RuntimeError("confusion_matrix: incomplete matrix reconstruction")

    return {
        "metric": "cost_ratio",
        "unit": "ratio",
        "row_labels": row_labels_sorted,
        "col_labels": col_labels_sorted,
        "values": matrix,
    }


def extract_eval_e2e_stats_table(tex_text: str) -> list[dict[str, Any]]:
    """
    Extract numeric rows from the LaTeX table `eval-e2e-stats.tex`.

    Parameters
    ----------
    tex_text : str
        Raw LaTeX table.

    Returns
    -------
    list[dict[str, Any]]
        Structured rows (p50/p75/p90 + experiment time).
    """

    rows: list[dict[str, Any]] = []
    for line in tex_text.splitlines():
        if "&" not in line or "\\\\" not in line:
            continue
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith("%"):
            continue
        parts = [p.strip() for p in line.split("&")]
        if len(parts) < 12:
            continue
        dataset = parts[0]
        if dataset.startswith("\\") or "TP-" not in dataset:
            continue
        parts[-1] = parts[-1].split("\\\\", 1)[0].strip()
        values = parts[1:]
        percentiles = ["p50", "p75", "p90"]
        for idx, percentile in enumerate(percentiles):
            merlin = float(values[idx * 3])
            actual = float(values[idx * 3 + 1])
            err_percent = values[idx * 3 + 2].replace("\\%", "%")
            rows.append(
                {
                    "dataset": dataset,
                    "percentile": percentile,
                    "merlin": merlin,
                    "actual": actual,
                    "error_percent": err_percent,
                    "unit": "s",
                }
            )
        rows.append(
            {
                "dataset": dataset,
                "metric": "experiment_time",
                "merlin_minutes": float(values[9]),
                "deployment_minutes": float(values[10]),
                "unit": "min",
            }
        )

    if not rows:
        raise RuntimeError("eval-e2e-stats.tex: no rows extracted")
    return rows


def build_common_metadata(pdf_path: Path, out_path: Path) -> dict[str, Any]:
    """
    Build common metadata for output JSON.

    Parameters
    ----------
    pdf_path : Path
        Source PDF.
    out_path : Path
        Output JSON path.

    Returns
    -------
    dict[str, Any]
        Metadata dict.
    """

    return {
        "source_pdf": str(pdf_path),
        "pdf_bytes": pdf_path.stat().st_size,
        "pdf_sha256": compute_sha256(pdf_path),
        "output_json": str(out_path),
    }


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("extern/tracked/vidur/paper/tex/graphs"),
        help="Directory containing Vidur paper graph PDFs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("context/summaries/vidur-kb/paper-results"),
        help="Directory to write extracted JSON tables.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("tmp/vidur_kb/paper_results_extract"),
        help="Temporary directory for intermediate SVG files.",
    )
    args = parser.parse_args()

    pdf_dir: Path = args.pdf_dir
    out_dir: Path = args.out_dir
    tmp_dir: Path = args.tmp_dir
    svg_dir = tmp_dir / "svg"

    ensure_dir(svg_dir)
    ensure_dir(out_dir)

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No PDFs found under {pdf_dir}")

    series_by_fill = {
        "real": "rgb(13.18512%, 30.831909%, 87.597656%)",
        "predicted": "rgb(87.5%, 48.970032%, 12.5%)",
    }
    fidelity_models = [
        "LLaMA2-7B (TP1)",
        "InternLM-20B (TP2)",
        "LLaMA2-70B (TP4)",
        "Qwen-72B (TP4)",
    ]

    bar_y_max: dict[str, float] = {
        "static_fidelity_v12_request_execution_plus_preemption_time_normalized_p50": 0.6,
        "static_fidelity_v12_request_execution_plus_preemption_time_normalized_p95": 0.8,
        "dynamic_fidelity_v8_request_e2e_time_normalized_75_p50": 0.10,
        "dynamic_fidelity_v8_request_e2e_time_normalized_75_p95": 0.20,
        "dynamic_fidelity_v8_request_e2e_time_normalized_85_p50": 0.15,
        "dynamic_fidelity_v8_request_e2e_time_normalized_85_p95": 0.20,
        "dynamic_fidelity_v8_request_e2e_time_normalized_95_p50": 0.15,
        "dynamic_fidelity_v8_request_e2e_time_normalized_95_p95": 0.30,
    }

    manifest_entries: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        basename = pdf_path.stem
        out_path = out_dir / f"{basename}.json"
        entry = build_common_metadata(pdf_path, out_path)

        payload: dict[str, Any]
        extraction: dict[str, Any]

        if basename in bar_y_max:
            svg_path = svg_dir / f"{basename}.svg"
            pdf_to_svg(pdf_path, svg_path)

            workload = "static" if basename.startswith("static_") else "dynamic"
            percentile = "p50" if basename.endswith("p50") else "p95"
            load_frac: float | None = None
            if workload == "dynamic":
                m = re.search(r"_normalized_(\d+)_", basename)
                if m:
                    load_frac = float(m.group(1)) / 100.0

            metric = (
                "request_execution_plus_preemption_time_normalized"
                if workload == "static"
                else "request_e2e_time_normalized"
            )
            payload = extract_bar_chart(
                svg_path=svg_path,
                y_max=bar_y_max[basename],
                model_labels=fidelity_models,
                metric=metric,
                unit="s/token",
                series_by_fill=series_by_fill,
                extra_metadata={
                    "workload": workload,
                    "percentile": percentile,
                    "load_frac_of_capacity": load_frac,
                },
            )
            extraction = {"method": "pdftocairo_svg_bar_digitize"}

        elif basename == "dynamic_fidelity_v8_request_e2e_time_normalized_error_trends_p95":
            svg_path = svg_dir / f"{basename}.svg"
            pdf_to_svg(pdf_path, svg_path)
            payload = extract_error_trends_p95(svg_path)
            extraction = {"method": "pdftocairo_svg_line_digitize"}

        elif basename == "parallel_coord":
            text = run_stdout(["pdftotext", str(pdf_path), "-"])
            payload = {
                "figure_type": "text_table",
                "metric": "qps_per_dollar",
                "unit": "QPS/$",
                "rows": extract_parallel_coord_qps(text),
            }
            extraction = {"method": "pdftotext"}

        elif basename == "capacity_per_dollar":
            parallel_pdf = pdf_dir / "parallel_coord.pdf"
            parallel_text = run_stdout(["pdftotext", str(parallel_pdf), "-"])
            parallel_rows = extract_parallel_coord_qps(parallel_text)
            payload = {
                "figure_type": "derived_text_table",
                "metric": "qps_per_dollar",
                "unit": "QPS/$",
                "rows": extract_capacity_per_dollar_from_parallel_coord(parallel_rows),
                "data_source_pdf": str(parallel_pdf),
                "notes": "Values extracted from parallel_coord.pdf text; capacity_per_dollar.pdf does not label bar values.",
            }
            extraction = {"method": "pdftotext_parallel_coord"}

        elif basename in (
            "llama70b_Chat1M_ttft_tbt_90_99_2.0_0.2",
            "qwen_ArxivSum_ttft_tbt_90_99_2.0_0.2",
        ):
            text = run_stdout(["pdftotext", str(pdf_path), "-"])
            payload = {"figure_type": "best_config_text", **extract_best_config(text)}
            extraction = {"method": "pdftotext"}

        elif basename == "vidur-hld":
            text = run_stdout(["pdftotext", str(pdf_path), "-"])
            payload = {"figure_type": "diagram_text", **extract_vidur_hld(text)}
            extraction = {"method": "pdftotext"}

        elif basename == "confusion_matrix_Llama-2-70b-hf":
            bbox_html = run_stdout(["pdftotext", "-bbox", str(pdf_path), "-"])
            payload = {"figure_type": "matrix", **extract_confusion_matrix_bbox(bbox_html)}
            extraction = {"method": "pdftotext_bbox"}

        elif basename == "fidelity_runtimes":
            table_path = Path("extern/tracked/vidur/paper/tex/tables/eval-e2e-stats.tex")
            payload = {
                "figure_type": "latex_table",
                "metric": "request_execution_time",
                "rows": extract_eval_e2e_stats_table(table_path.read_text(encoding="utf-8")),
                "data_source_tex": str(table_path),
            }
            extraction = {"method": "latex_parse_eval-e2e-stats"}

        else:
            payload = {
                "figure_type": "unextracted",
                "notes": "No numeric table extracted for this figure (not required for fidelity metric comparisons).",
            }
            extraction = {"method": "none"}

        payload_full = {**entry, "extraction": extraction, **payload}
        write_json(out_path, payload_full)

        entry["output_json_bytes"] = out_path.stat().st_size
        entry["output_json_sha256"] = compute_sha256(out_path)
        entry["status"] = "ok" if payload.get("figure_type") != "unextracted" else "skipped"
        entry["extraction"] = extraction
        manifest_entries.append(entry)

    manifest = {
        "source_pdf_dir": str(pdf_dir),
        "output_dir": str(out_dir),
        "entries": manifest_entries,
    }
    write_json(out_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
