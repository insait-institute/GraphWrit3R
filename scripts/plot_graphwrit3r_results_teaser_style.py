#!/usr/bin/env python3
"""Generate GraphWrit3R website result charts in the teaser style.

The 2x3 figure contains:
  1. Object Recall
  2. Predicate Recall
  3. Triplet Recall
  4. ScanNet20 Object Detection: Average
  5. ScanNet20 Object Detection: Chair
  6. ScanNet20 Object Detection: Table

Outputs:
    graphwrit3r_results_teaser_style.svg
    graphwrit3r_results_teaser_style.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
RECALL_ROWS = [
    {
        "method": "Qwen3-VL-32B",
        "obj_R5": 0.26,
        "obj_R10": 0.31,
        "pred_R3": 0.58,
        "pred_R5": 0.59,
        "trip_R50": 0.42,
        "trip_R100": 0.45,
    },
    {
        "method": "GPT-5.4",
        "obj_R5": 0.36,
        "obj_R10": 0.47,
        "pred_R3": 0.78,
        "pred_R5": 0.79,
        "trip_R50": 0.63,
        "trip_R100": 0.66,
    },
    {
        "method": "ConceptGraphs",
        "obj_R5": 0.37,
        "obj_R10": 0.46,
        "pred_R3": 0.74,
        "pred_R5": 0.79,
        "trip_R50": 0.69,
        "trip_R100": 0.71,
    },
    {
        "method": "Open3DSG",
        "obj_R5": 0.56,
        "obj_R10": 0.61,
        "pred_R3": 0.58,
        "pred_R5": 0.65,
        "trip_R50": 0.55,
        "trip_R100": 0.56,
    },
    {
        "method": "RelationField",
        "obj_R5": 0.69,
        "obj_R10": 0.80,
        "pred_R3": 0.76,
        "pred_R5": 0.82,
        "trip_R50": 0.73,
        "trip_R100": 0.74,
    },
    {
        "method": "ReLaGS",
        "obj_R5": 0.68,
        "obj_R10": 0.79,
        "pred_R3": 0.79,
        "pred_R5": 0.87,
        "trip_R50": np.nan,
        "trip_R100": np.nan,
    },
    {
        "method": "GraphWrit3R (ours)",
        "obj_R5": 0.69,
        "obj_R10": 0.76,
        "pred_R3": 0.85,
        "pred_R5": 0.87,
        "trip_R50": 0.74,
        "trip_R100": 0.79,
    },
]

RECALL_METHOD_ORDER = [
    "Qwen3-VL-32B",
    "GPT-5.4",
    "ConceptGraphs",
    "Open3DSG",
    "RelationField",
    "ReLaGS",
    "GraphWrit3R (ours)",
]

RECALL_PANELS = [
    {
        "title": "Object Recall",
        "metrics": [("R@5", "obj_R5"), ("R@10", "obj_R10")],
    },
    {
        "title": "Predicate Recall",
        "metrics": [("R@3", "pred_R3"), ("R@5", "pred_R5")],
    },
    {
        "title": "Triplet Recall",
        "metrics": [("R@50", "trip_R50"), ("R@100", "trip_R100")],
    },
]

RECALL_BY_METHOD = {row["method"]: row for row in RECALL_ROWS}

DETECTION_ROWS = [
    {
        "method": "V-DETR",
        "Average": 65.7,
        "Chair": 82.5,
        "Cabinet": 44.4,
        "Table": 55.3,
    },
    {
        "method": "SceneScript",
        "Average": 49.5,
        "Chair": 81.1,
        "Cabinet": 30.3,
        "Table": 55.8,
    },
    {
        "method": "SpatialLM",
        "Average": 66.2,
        "Chair": 86.6,
        "Cabinet": 40.0,
        "Table": 63.9,
    },
    {
        "method": "GraphWrit3R (ours)",
        "Average": 68.9,
        "Chair": 85.7,
        "Cabinet": 52.4,
        "Table": 68.3,
    },
]

DETECTION_METHOD_ORDER = [
    "V-DETR",
    "SceneScript",
    "SpatialLM",
    "GraphWrit3R (ours)",
]

# Overall performance plus the two most frequent classes in the original
# selection.
DETECTION_METRICS = ["Average", "Chair", "Table"]

DETECTION_BY_METHOD = {row["method"]: row for row in DETECTION_ROWS}


# -----------------------------------------------------------------------------
# Teaser-inspired styling
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MethodStyle:
    marker: str
    color: str
    short_label: str


METHOD_STYLES = {
    "Qwen3-VL-32B": MethodStyle("D", "#F08E90", "Qwen3-VL"),
    "GPT-5.4": MethodStyle("8", "#C9C9C9", "GPT-5.4"),
    "ConceptGraphs": MethodStyle("p", "#CDA5D9", "ConceptGraphs"),
    "Open3DSG": MethodStyle("s", "#86D9D0", "Open3DSG"),
    "RelationField": MethodStyle("o", "#8EB7F0", "RelationField"),
    "ReLaGS": MethodStyle("^", "#8FCF62", "ReLaGS"),
    "V-DETR": MethodStyle("o", "#B7BDC7", "V-DETR"),
    "SceneScript": MethodStyle("s", "#929CAA", "SceneScript"),
    "SpatialLM": MethodStyle("D", "#66758A", "SpatialLM"),
    "GraphWrit3R (ours)": MethodStyle("*", "#63C56E", "GraphWrit3R\n(ours)"),
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "svg.fonttype": "none",
    }
)

TEXT = "#161A22"
MUTED = "#4F5663"
AXIS = "#252B35"
GRID = "#D4D8DF"
BG = "#FFFFFF"
OURS_TEXT = "#2D8E39"

Y_MIN = 20.0
Y_MAX = 90.0
Y_PLOT_MIN = 12.0
Y_PLOT_MAX = 94.0
Y_TICKS = (20, 40, 60, 80)

MARKER_SIZE = 125
PAIR_OFFSET = 0.15


def round_marker_corners(marker_artist) -> None:
    """Match the softened marker joins used by the teaser figure."""
    if hasattr(marker_artist, "set_joinstyle"):
        marker_artist.set_joinstyle("round")


def draw_teaser_axes(
    ax,
    *,
    count: int,
    ylabel: str,
    show_y_ticklabels: bool,
) -> None:
    """Draw the heavy L-shaped axes and restrained horizontal guides."""
    x_min = -0.55
    x_max = count - 0.45

    ax.set_xlim(x_min - 0.18, x_max + 0.08)
    ax.set_ylim(Y_PLOT_MIN, Y_PLOT_MAX)
    ax.axis("off")

    ax.plot(
        [x_min, x_max],
        [Y_MIN, Y_MIN],
        color=AXIS,
        linewidth=1.8,
        zorder=10,
    )
    ax.plot(
        [x_min, x_min],
        [Y_MIN, Y_MAX],
        color=AXIS,
        linewidth=1.8,
        zorder=10,
    )

    for tick in Y_TICKS[1:]:
        ax.plot(
            [x_min, x_max],
            [tick, tick],
            color=GRID,
            linewidth=0.9,
            zorder=1,
        )

    if show_y_ticklabels:
        for tick in Y_TICKS:
            ax.text(
                x_min - 0.13,
                tick,
                str(tick),
                ha="right",
                va="center",
                fontsize=9.0,
                fontweight="semibold",
                color=MUTED,
            )

        ax.text(
            x_min - 0.50,
            (Y_MIN + Y_MAX) / 2,
            ylabel,
            ha="center",
            va="center",
            rotation=90,
            fontsize=11.5,
            fontweight="bold",
            color=AXIS,
        )


def draw_method_marker(
    ax,
    *,
    x: float,
    y: float,
    method: str,
    filled: bool,
) -> None:
    """Draw one fixed-size method marker."""
    style = METHOD_STYLES[method]

    marker_artist = ax.scatter(
        x,
        y,
        s=MARKER_SIZE,
        marker=style.marker,
        facecolors=style.color if filled else BG,
        edgecolors=style.color,
        linewidths=2.0 if not filled else 1.0,
        zorder=5,
    )
    round_marker_corners(marker_artist)


def add_value_label(
    ax,
    *,
    x: float,
    value: float,
    method: str,
    decimals: int,
    is_best: bool,
) -> None:
    """Write a compact value label with a white readability stroke."""
    is_ours = method == "GraphWrit3R (ours)"
    ax.text(
        x,
        value + 2.35,
        f"{value:.{decimals}f}",
        ha="center",
        va="bottom",
        fontsize=8.6,
        fontweight="bold" if is_best or is_ours else "semibold",
        color=OURS_TEXT if is_ours else TEXT,
        path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)],
        zorder=7,
    )


def draw_method_ticks(ax, methods: Sequence[str]) -> None:
    """Place compact method labels under the manual x-axis."""
    for index, method in enumerate(methods):
        style = METHOD_STYLES[method]
        is_ours = method == "GraphWrit3R (ours)"
        ax.text(
            index,
            Y_MIN - 3.0,
            style.short_label,
            ha="right",
            va="top",
            rotation=31,
            rotation_mode="anchor",
            fontsize=7.8,
            fontweight="bold" if is_ours else "semibold",
            color=OURS_TEXT if is_ours else TEXT,
        )


def draw_recall_panel(
    ax,
    panel: dict,
    *,
    show_y_axis: bool,
) -> None:
    """Draw a paired open/filled marker panel for one recall family."""
    methods = RECALL_METHOD_ORDER
    (label_a, col_a), (label_b, col_b) = panel["metrics"]
    values_a = np.asarray(
        [100.0 * RECALL_BY_METHOD[method][col_a] for method in methods],
        dtype=float,
    )
    values_b = np.asarray(
        [100.0 * RECALL_BY_METHOD[method][col_b] for method in methods],
        dtype=float,
    )
    best_a = np.nanmax(values_a)
    best_b = np.nanmax(values_b)

    draw_teaser_axes(
        ax,
        count=len(methods),
        ylabel="Recall (%)",
        show_y_ticklabels=show_y_axis,
    )

    for index, method in enumerate(methods):
        value_a = values_a[index]
        value_b = values_b[index]

        if np.isnan(value_a) and np.isnan(value_b):
            ax.text(
                index,
                Y_MIN + 2.0,
                "N/R",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontstyle="italic",
                fontweight="semibold",
                color=MUTED,
            )
            continue

        x_a = index - PAIR_OFFSET
        x_b = index + PAIR_OFFSET
        style = METHOD_STYLES[method]

        if not np.isnan(value_a) and not np.isnan(value_b):
            ax.plot(
                [x_a, x_b],
                [value_a, value_b],
                color=style.color,
                linewidth=2.0,
                alpha=0.62,
                zorder=3,
            )

        if not np.isnan(value_a):
            draw_method_marker(
                ax,
                x=x_a,
                y=value_a,
                method=method,
                filled=False,
            )
            add_value_label(
                ax,
                x=x_a,
                value=value_a,
                method=method,
                decimals=0,
                is_best=np.isclose(value_a, best_a),
            )

        if not np.isnan(value_b):
            draw_method_marker(
                ax,
                x=x_b,
                y=value_b,
                method=method,
                filled=True,
            )
            add_value_label(
                ax,
                x=x_b,
                value=value_b,
                method=method,
                decimals=0,
                is_best=np.isclose(value_b, best_b),
            )

    draw_method_ticks(ax, methods)
    ax.set_title(
        f"{panel['title']} — {label_a} / {label_b}",
        loc="left",
        fontsize=15.5,
        fontweight="bold",
        color=TEXT,
        pad=12,
    )


def draw_detection_panel(
    ax,
    metric: str,
    *,
    show_y_axis: bool,
) -> None:
    """Draw one ScanNet20 metric with teaser-style method markers."""
    methods = DETECTION_METHOD_ORDER
    values = np.asarray(
        [DETECTION_BY_METHOD[method][metric] for method in methods],
        dtype=float,
    )
    best_value = np.nanmax(values)

    draw_teaser_axes(
        ax,
        count=len(methods),
        ylabel="F1 Score (%)",
        show_y_ticklabels=show_y_axis,
    )

    for index, (method, value) in enumerate(zip(methods, values)):
        draw_method_marker(
            ax,
            x=float(index),
            y=value,
            method=method,
            filled=True,
        )
        add_value_label(
            ax,
            x=float(index),
            value=value,
            method=method,
            decimals=1,
            is_best=np.isclose(value, best_value),
        )

    draw_method_ticks(ax, methods)
    ax.set_title(
        f"ScanNet20 Detection — {metric}",
        loc="left",
        fontsize=15.5,
        fontweight="bold",
        color=TEXT,
        pad=12,
    )


def make_plot(output_dir: Union[str, Path] = ".") -> None:
    """Render and save the 2x3 website figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17.0, 10.2),
        facecolor=BG,
        gridspec_kw={"hspace": 0.42, "wspace": 0.22},
    )

    for index, (ax, panel) in enumerate(zip(axes[0], RECALL_PANELS)):
        draw_recall_panel(ax, panel, show_y_axis=index == 0)

    for index, metric in enumerate(DETECTION_METRICS):
        draw_detection_panel(
            axes[1, index],
            metric,
            show_y_axis=index == 0,
        )

    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.94,
        bottom=0.08,
    )

    svg_path = output_dir / "graphwrit3r_results_teaser_style.svg"
    png_path = output_dir / "graphwrit3r_results_teaser_style.png"

    fig.savefig(svg_path, bbox_inches="tight", facecolor=BG)
    fig.savefig(png_path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    print(f"Saved: {svg_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    make_plot(Path(__file__).resolve().parent)
