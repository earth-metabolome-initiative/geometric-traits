#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


SVG_WIDTH = 1280
SVG_HEIGHT = 920
MARGIN_LEFT = 96
MARGIN_RIGHT = 60
PLOT_WIDTH = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT

RUNTIME_TOP = 118
RUNTIME_HEIGHT = 288
SPEEDUP_TOP = 468
SPEEDUP_HEIGHT = 288

TABLE_TOP = 792
TABLE_HEIGHT = 110

RUST_COLOR = "#d1495b"
RDKIT_COLOR = "#2f6690"
SPEEDUP_COLOR = "#3a7d44"
BG_COLOR = "#f8f7f4"
FG_COLOR = "#1d1d1f"
GRID_COLOR = "#ddd"


@dataclass
class TimingRow:
    name: str
    rust_s: float
    rdkit_s: float
    expected_bond_matches: int
    expected_similarity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the MCES 200k runtime summary JSON and histogram SVG."
    )
    parser.add_argument("--input", required=True, help="Path to the fresh timing JSON dump")
    parser.add_argument("--summary-output", required=True, help="Path to the summary JSON")
    parser.add_argument("--svg-output", required=True, help="Path to the SVG output")
    return parser.parse_args()


def load_rows(path: Path) -> list[TimingRow]:
    raw = json.loads(path.read_text())
    return [TimingRow(**row) for row in raw]


def pop_std(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def geo_ticks(min_value: float, max_value: float, count: int) -> list[float]:
    if min_value <= 0 or max_value <= 0:
        raise ValueError("geo_ticks requires positive values")
    if math.isclose(min_value, max_value):
        return [min_value] * count
    log_min = math.log10(min_value)
    log_max = math.log10(max_value)
    return [10 ** (log_min + (log_max - log_min) * index / (count - 1)) for index in range(count)]


def histogram(values: list[float], bins: int, min_value: float, max_value: float) -> tuple[list[float], list[int]]:
    log_min = math.log10(min_value)
    log_max = math.log10(max_value)
    edges = [10 ** (log_min + (log_max - log_min) * index / bins) for index in range(bins + 1)]
    counts = [0] * bins
    for value in values:
        if value <= min_value:
            counts[0] += 1
            continue
        if value >= max_value:
            counts[-1] += 1
            continue
        position = (math.log10(value) - log_min) / (log_max - log_min)
        index = min(bins - 1, max(0, int(position * bins)))
        counts[index] += 1
    return edges, counts


def x_from_value(value: float, min_value: float, max_value: float, left: float, width: float) -> float:
    log_min = math.log10(min_value)
    log_max = math.log10(max_value)
    return left + (math.log10(value) - log_min) / (log_max - log_min) * width


def y_from_count(count: int, max_count: int, top: float, height: float) -> float:
    if count <= 0:
        return top + height
    return top + height - (math.log10(count + 1) / math.log10(max_count + 1)) * height


def svg_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt_num(value: float) -> str:
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:.0f}"
    if abs_value >= 100:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if abs_value >= 10:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if abs_value >= 1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if abs_value >= 0.1:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.3g}"


def fmt_duration(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:.6f}s"
    millis = seconds * 1000
    if millis >= 1:
        return f"{millis:.3f}ms"
    micros = seconds * 1_000_000
    return f"{micros:.3f}µs"


def runtime_tick_label(ms: float) -> str:
    if ms >= 1000:
        exponent = int(math.floor(math.log10(ms)))
        mantissa = ms / (10 ** exponent)
        if mantissa >= 9.95:
            mantissa = 1.0
            exponent += 1
        if mantissa >= 9.5:
            return f"1e+{exponent}"
        if mantissa >= 1.5:
            return f"{mantissa:.1f}e+{exponent}"
        return f"{mantissa:.0f}e+{exponent}"
    return fmt_num(ms)


def ratio_tick_label(value: float) -> str:
    if math.isclose(value, 1.0):
        return "1"
    return fmt_num(value)


def build_summary(rows: list[TimingRow]) -> dict[str, object]:
    rust_ms = [row.rust_s * 1000 for row in rows]
    rdkit_ms = [row.rdkit_s * 1000 for row in rows]

    slowest_rust = max(rows, key=lambda row: row.rust_s)
    slowest_rdkit = max(rows, key=lambda row: row.rdkit_s)
    best_rust_advantage = max(rows, key=lambda row: row.rdkit_s / row.rust_s)
    worst_rust_disadvantage = max(rows, key=lambda row: row.rust_s / row.rdkit_s)

    def case_dict(row: TimingRow) -> dict[str, object]:
        rdkit_over_rust = row.rdkit_s / row.rust_s
        rust_over_rdkit = row.rust_s / row.rdkit_s
        return {
            "name": row.name,
            "rust_s": row.rust_s,
            "rdkit_s": row.rdkit_s,
            "expected_bond_matches": row.expected_bond_matches,
            "expected_similarity": row.expected_similarity,
            "ratio_rdkit_over_rust": rdkit_over_rust,
            "ratio_rust_over_rdkit": rust_over_rdkit,
        }

    return {
        "slowest_rust": case_dict(slowest_rust),
        "slowest_rdkit": case_dict(slowest_rdkit),
        "best_rust_advantage": case_dict(best_rust_advantage),
        "worst_rust_disadvantage": case_dict(worst_rust_disadvantage),
        "rust_total_s": sum(row.rust_s for row in rows),
        "rdkit_total_s": sum(row.rdkit_s for row in rows),
        "rust_mean_ms": statistics.fmean(rust_ms),
        "rust_median_ms": statistics.median(rust_ms),
        "rust_population_std_ms": pop_std(rust_ms),
        "rdkit_mean_ms": statistics.fmean(rdkit_ms),
        "rdkit_median_ms": statistics.median(rdkit_ms),
        "rdkit_population_std_ms": pop_std(rdkit_ms),
        "rust_faster_cases": sum(1 for row in rows if row.rust_s < row.rdkit_s),
        "rdkit_faster_cases": sum(1 for row in rows if row.rdkit_s < row.rust_s),
    }


def add_runtime_histogram(svg: list[str], rows: list[TimingRow]) -> None:
    left = MARGIN_LEFT
    top = RUNTIME_TOP
    height = RUNTIME_HEIGHT

    rust_ms = [row.rust_s * 1000 for row in rows]
    rdkit_ms = [row.rdkit_s * 1000 for row in rows]
    all_ms = rust_ms + rdkit_ms
    min_ms = min(value for value in all_ms if value > 0)
    max_ms = max(all_ms)
    bins = 48
    edges, rust_counts = histogram(rust_ms, bins, min_ms, max_ms)
    _, rdkit_counts = histogram(rdkit_ms, bins, min_ms, max_ms)
    max_count = max(max(rust_counts), max(rdkit_counts))

    svg.append(f'<text x="{left + PLOT_WIDTH / 2:.1f}" y="108" text-anchor="middle" font-size="16" font-weight="700">Per-case runtime distribution</text>')
    svg.append(
        f'<rect x="{left}" y="{top}" width="{PLOT_WIDTH}" height="{height}" fill="white" stroke="#222" stroke-width="1"/>'
    )

    y_ticks = [0, 10, 100, 1000, max_count]
    for tick in y_ticks:
        y = y_from_count(tick, max_count, top, height)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + PLOT_WIDTH}" y2="{y:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        svg.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#444">{tick}</text>')

    x_ticks = geo_ticks(min_ms, max_ms, 5)
    for tick in x_ticks:
        x = x_from_value(tick, min_ms, max_ms, left, PLOT_WIDTH)
        svg.append(f'<line x1="{x:.2f}" y1="{top + height}" x2="{x:.2f}" y2="{top + height + 6}" stroke="#222" stroke-width="1"/>')
        svg.append(f'<text x="{x:.2f}" y="{top + height + 22}" text-anchor="middle" font-size="11" fill="#444">{runtime_tick_label(tick)}</text>')

    svg.append(
        f'<text x="{left + PLOT_WIDTH / 2:.1f}" y="{top + height + 38}" text-anchor="middle" font-size="12" fill="#444">runtime (ms) (log bins)</text>'
    )
    svg.append(
        f'<text x="54" y="{top + height / 2:.1f}" text-anchor="middle" font-size="12" fill="#444" transform="rotate(-90 54,{top + height / 2:.1f})">count (log-scaled)</text>'
    )

    for index in range(bins):
        x0 = x_from_value(edges[index], min_ms, max_ms, left, PLOT_WIDTH)
        x1 = x_from_value(edges[index + 1], min_ms, max_ms, left, PLOT_WIDTH)
        width = x1 - x0
        bar_width = width * 0.48
        rust_y = y_from_count(rust_counts[index], max_count, top, height)
        rdkit_y = y_from_count(rdkit_counts[index], max_count, top, height)
        svg.append(
            f'<rect x="{x0:.2f}" y="{rust_y:.2f}" width="{bar_width:.2f}" height="{top + height - rust_y:.2f}" fill="{RUST_COLOR}" fill-opacity="0.55"/>'
        )
        svg.append(
            f'<rect x="{x0 + width * 0.52:.2f}" y="{rdkit_y:.2f}" width="{bar_width:.2f}" height="{top + height - rdkit_y:.2f}" fill="{RDKIT_COLOR}" fill-opacity="0.55"/>'
        )

    legend_x = left + PLOT_WIDTH - 200
    legend_y = top + 14
    svg.append(f'<rect x="{legend_x}" y="{legend_y}" width="180" height="48" rx="5" fill="#ffffff" fill-opacity="0.88" stroke="#d9d6cf" stroke-width="1"/>')
    svg.append(f'<rect x="{legend_x + 12}" y="{legend_y + 16}" width="12" height="12" fill="{RUST_COLOR}" fill-opacity="0.55"/>')
    svg.append(f'<text x="{legend_x + 30}" y="{legend_y + 26}" font-size="12" fill="{FG_COLOR}">Rust</text>')
    svg.append(f'<rect x="{legend_x + 12}" y="{legend_y + 36}" width="12" height="12" fill="{RDKIT_COLOR}" fill-opacity="0.55"/>')
    svg.append(f'<text x="{legend_x + 30}" y="{legend_y + 46}" font-size="12" fill="{FG_COLOR}">RDKit</text>')


def add_speedup_histogram(svg: list[str], rows: list[TimingRow]) -> None:
    left = MARGIN_LEFT
    top = SPEEDUP_TOP
    height = SPEEDUP_HEIGHT

    ratios = [row.rdkit_s / row.rust_s for row in rows]
    min_ratio = min(value for value in ratios if value > 0)
    max_ratio = max(ratios)
    bins = 48
    edges, counts = histogram(ratios, bins, min_ratio, max_ratio)
    max_count = max(counts)

    svg.append(f'<text x="{left + PLOT_WIDTH / 2:.1f}" y="458" text-anchor="middle" font-size="16" font-weight="700">Per-case speedup distribution</text>')
    svg.append(
        f'<rect x="{left}" y="{top}" width="{PLOT_WIDTH}" height="{height}" fill="white" stroke="#222" stroke-width="1"/>'
    )

    y_ticks = [0, 10, 100, 1000, max_count]
    for tick in y_ticks:
        y = y_from_count(tick, max_count, top, height)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + PLOT_WIDTH}" y2="{y:.2f}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        svg.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#444">{tick}</text>')

    x_ticks = []
    if min_ratio < 1 < max_ratio:
        x_ticks = [min_ratio, math.sqrt(min_ratio), 1.0, math.sqrt(max_ratio), max_ratio]
    else:
        x_ticks = geo_ticks(min_ratio, max_ratio, 5)
    for tick in x_ticks:
        x = x_from_value(tick, min_ratio, max_ratio, left, PLOT_WIDTH)
        svg.append(f'<line x1="{x:.2f}" y1="{top + height}" x2="{x:.2f}" y2="{top + height + 6}" stroke="#222" stroke-width="1"/>')
        svg.append(f'<text x="{x:.2f}" y="{top + height + 22}" text-anchor="middle" font-size="11" fill="#444">{ratio_tick_label(tick)}</text>')

    svg.append(
        f'<text x="{left + PLOT_WIDTH / 2:.1f}" y="{top + height + 34}" text-anchor="middle" font-size="12" fill="#444">speedup factor (RDKit / Rust, log bins)</text>'
    )
    svg.append(
        f'<text x="54" y="{top + height / 2:.1f}" text-anchor="middle" font-size="12" fill="#444" transform="rotate(-90 54,{top + height / 2:.1f})">count (log-scaled)</text>'
    )

    for index in range(bins):
        x0 = x_from_value(edges[index], min_ratio, max_ratio, left, PLOT_WIDTH)
        x1 = x_from_value(edges[index + 1], min_ratio, max_ratio, left, PLOT_WIDTH)
        y = y_from_count(counts[index], max_count, top, height)
        svg.append(
            f'<rect x="{x0:.2f}" y="{y:.2f}" width="{x1 - x0:.2f}" height="{top + height - y:.2f}" fill="{SPEEDUP_COLOR}" fill-opacity="0.55"/>'
        )

    if min_ratio < 1 < max_ratio:
        x = x_from_value(1.0, min_ratio, max_ratio, left, PLOT_WIDTH)
        svg.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + height}" stroke="#777" stroke-width="2" stroke-dasharray="6 4"/>'
        )

    legend_x = left + PLOT_WIDTH - 200
    legend_y = top + 14
    svg.append(f'<rect x="{legend_x}" y="{legend_y}" width="180" height="48" rx="5" fill="#ffffff" fill-opacity="0.88" stroke="#d9d6cf" stroke-width="1"/>')
    svg.append(f'<rect x="{legend_x + 12}" y="{legend_y + 16}" width="12" height="12" fill="{SPEEDUP_COLOR}" fill-opacity="0.55"/>')
    svg.append(f'<text x="{legend_x + 30}" y="{legend_y + 26}" font-size="12" fill="{FG_COLOR}">RDKit / Rust</text>')
    svg.append(f'<line x1="{legend_x + 12}" y1="{legend_y + 36}" x2="{legend_x + 24}" y2="{legend_y + 36}" stroke="#777" stroke-width="2" stroke-dasharray="6 4"/>')
    svg.append(f'<text x="{legend_x + 30}" y="{legend_y + 40}" font-size="12" fill="{FG_COLOR}">parity = 1x</text>')


def add_summary_table(svg: list[str], summary: dict[str, object]) -> None:
    table_x = 60
    table_width = 1160
    svg.append(f'<rect x="{table_x}" y="{TABLE_TOP}" width="{table_width}" height="{TABLE_HEIGHT}" rx="6" fill="#ffffff" stroke="#d9d6cf" stroke-width="1"/>')
    svg.append(f'<rect x="{table_x}" y="{TABLE_TOP}" width="{table_width}" height="24" rx="6" fill="#ece8de"/>')
    for y in (816, 838, 860, 882):
        stroke = "#d9d6cf" if y == 816 else "#e6e2d8"
        svg.append(f'<line x1="{table_x}" y1="{y}" x2="{table_x + table_width}" y2="{y}" stroke="{stroke}" stroke-width="1"/>')
    for x in (318, 600, 760, 930):
        svg.append(f'<line x1="{x}" y1="{TABLE_TOP}" x2="{x}" y2="{TABLE_TOP + TABLE_HEIGHT}" stroke="#e6e2d8" stroke-width="1"/>')

    headers = [("Category", 72), ("Case", 330), ("Rust", 612), ("RDKit", 772), ("Ratio", 942)]
    for label, x in headers:
        svg.append(f'<text x="{x}" y="808" font-size="13" font-weight="700" fill="{FG_COLOR}">{label}</text>')

    rows = [
        ("Slowest Rust", "slowest_rust", "rdkit_over_rust"),
        ("Slowest RDKit", "slowest_rdkit", "rdkit_over_rust"),
        ("Best Rust advantage", "best_rust_advantage", "rdkit_over_rust"),
        ("Worst Rust disadvantage", "worst_rust_disadvantage", "rust_over_rdkit"),
    ]
    row_y = [831, 853, 875, 897]
    for (label, key, ratio_mode), y in zip(rows, row_y, strict=True):
        case = summary[key]
        assert isinstance(case, dict)
        ratio = case["ratio_rdkit_over_rust"] if ratio_mode == "rdkit_over_rust" else case["ratio_rust_over_rdkit"]
        ratio_label = "RDKit/Rust" if ratio_mode == "rdkit_over_rust" else "Rust/RDKit"
        svg.append(f'<text x="72" y="{y}" font-size="13" font-weight="700" fill="{FG_COLOR}">{label}</text>')
        svg.append(f'<text x="330" y="{y}" font-size="13" fill="{FG_COLOR}">{svg_escape(str(case["name"]))}</text>')
        svg.append(f'<text x="612" y="{y}" font-size="13" fill="{FG_COLOR}">{fmt_duration(float(case["rust_s"]))}</text>')
        svg.append(f'<text x="772" y="{y}" font-size="13" fill="{FG_COLOR}">{fmt_duration(float(case["rdkit_s"]))}</text>')
        svg.append(f'<text x="942" y="{y}" font-size="13" fill="{FG_COLOR}">{ratio_label} {float(ratio):.2f}x</text>')


def build_svg(rows: list[TimingRow], summary: dict[str, object]) -> str:
    rust_mean_ms = float(summary["rust_mean_ms"])
    rust_std_ms = float(summary["rust_population_std_ms"])
    rust_median_ms = float(summary["rust_median_ms"])
    rdkit_mean_ms = float(summary["rdkit_mean_ms"])
    rdkit_std_ms = float(summary["rdkit_population_std_ms"])
    rdkit_median_ms = float(summary["rdkit_median_ms"])

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" style="font-family: sans-serif;">',
        f'<rect width="100%" height="100%" fill="{BG_COLOR}"/>',
        f'<text x="640" y="36" text-anchor="middle" font-size="24" font-weight="700" fill="{FG_COLOR}">MCES 200k runtime comparison</text>',
        '<text x="640" y="62" text-anchor="middle" font-size="14" fill="#555">Rust timings come from a fresh sequential parity run on the 200k corpus; RDKit timings are the per-case reference times recorded when that same corpus was generated.</text>',
        (
            f'<text x="640" y="84" text-anchor="middle" font-size="13">'
            f'<tspan fill="{RUST_COLOR}">Rust mean ± std: {rust_mean_ms:.2f} ± {rust_std_ms:.2f} ms; median: {rust_median_ms:.2f} ms</tspan>'
            f'<tspan fill="#666"> | </tspan>'
            f'<tspan fill="{RDKIT_COLOR}">RDKit mean ± std: {rdkit_mean_ms:.2f} ± {rdkit_std_ms:.2f} ms; median: {rdkit_median_ms:.2f} ms</tspan>'
            f'</text>'
        ),
    ]

    add_runtime_histogram(svg, rows)
    add_speedup_histogram(svg, rows)
    add_summary_table(svg, summary)
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input))
    if len(rows) != 200000:
        raise SystemExit(f"expected 200000 timing rows, got {len(rows)}")
    summary = build_summary(rows)

    summary_output = Path(args.summary_output)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n")

    svg_output = Path(args.svg_output)
    svg_output.write_text(build_svg(rows, summary))


if __name__ == "__main__":
    main()
