#!/usr/bin/env python3
"""Draw the three-panel architecture schematic used as Figure 1.

The figure is an illustrative eight-node chain with damping on the final four
coordinates. The same state coordinates and damping support are drawn in the
free and both nudged phases. This script does not read experiment results.

Usage:
    python figure_architecture_overview_source.py --output-dir paper_figures

Outputs:
    figure_architecture_overview.png  (2048 x 684 pixels, 160 dpi)
    figure_architecture_overview.pdf  (vector figure)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

W, H = 2048, 684
INK = "#1c242e"
LIGHT_BLUE = "#f0f6ff"
BLUE = "#6989b4"
RED = "#d95562"
PURPLE = "#9461da"


def arrow(ax, start, end, *, size=16, lw=1.5, color=INK):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=size,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0,
        capstyle="round", joinstyle="round",
    ))


def rounded(ax, x, y, width, height, *, fill=LIGHT_BLUE, edge="none",
            linewidth=1.2, radius=14, dash=None):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=linewidth,
        linestyle=dash if dash is not None else "solid",
    )
    ax.add_patch(box)
    return box


def centered(ax, x, y, text, *, size=13, weight="normal", color=INK, **kwargs):
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            fontweight=weight, color=color, **kwargs)


def chain(ax, x_first, x_last, y, *, node_radius=8, support_box=True):
    xs = [x_first + j * (x_last - x_first) / 7 for j in range(8)]
    ax.plot([xs[0], xs[-1]], [y, y], color=INK, lw=2.0,
            solid_capstyle="round", zorder=2)
    if support_box:
        rounded(ax, xs[4]-31, y-26, xs[-1]-xs[4]+64, 84,
                fill="#f7faff", edge=BLUE, linewidth=1.05, radius=5,
                dash=(0, (6, 4)))
    for j in range(4, 8):
        arrow(ax, (xs[j], y+12), (xs[j], y+46), size=14, lw=1.35)
    for x in xs:
        ax.add_patch(plt.Circle((x, y), node_radius, facecolor="white",
                                edgecolor=INK, lw=1.8, zorder=6))
    return xs


def render_architecture_figure(output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "figure_architecture_overview.png"
    pdf = output_dir / "figure_architecture_overview.pdf"

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }):
        fig = plt.figure(figsize=(12.8, 4.275), dpi=160)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("auto")
        ax.axis("off")

        for x in (681, 1375):
            ax.plot([x, x], [54, 647], color="#97accd", lw=1.0)
        ax.text(40, 49, "(a)  Physical architecture", fontsize=17.5,
                fontweight="bold", color=INK, va="center")
        ax.text(720, 49, "(b)  Three relaxation phases", fontsize=17.5,
                fontweight="bold", color=INK, va="center")
        ax.text(1415, 49, "(c)  Local parameter credit", fontsize=17.5,
                fontweight="bold", color=INK, va="center")

        # An illustrative eight-node chain with four damped readout-side nodes.
        chain(ax, 78, 610, 277)
        ax.text(38, 146, "input", fontsize=13, color=INK, va="center")
        arrow(ax, (78, 177), (78, 253), size=16)
        centered(ax, 612, 147, "readout", size=13)
        arrow(ax, (612, 176), (612, 251), size=16)
        ax.plot([157, 528], [208, 208], color=BLUE, lw=1.35)
        ax.plot([157, 157], [190, 228], color=BLUE, lw=1.35)
        ax.plot([528, 528], [190, 228], color=BLUE, lw=1.35)
        centered(ax, 343, 185, "conservative interior", size=12.2)
        centered(ax, 498, 384, "localized damping\nsupport $B$", size=13)
        centered(ax, 347, 482,
                 r"$\mathcal{P}_{\mathrm{diss}}=\|R^{1/2}B\dot q\|^2$", size=17.0)
        rounded(ax, 40, 551, 618, 69, fill=LIGHT_BLUE)
        centered(ax, 349, 585, "conservative transport + localized relaxation",
                 size=12.0)

        # The same chain and damping support in the three relaxation phases.
        phases = [
            ("free", r"$\beta=0$", r"$q_0$", 162, "#edf4ff"),
            ("positive", r"$+\beta$", r"$q_{+\beta}$", 308, "#fff0f1"),
            ("negative", r"$-\beta$", r"$q_{-\beta}$", 454, "#f6edff"),
        ]
        for name, beta, q_label, y, fill in phases:
            rounded(ax, 699, y-37, 143, 73, fill=fill, radius=9)
            centered(ax, 769, y-13, name, size=12.6, weight="bold")
            centered(ax, 769, y+17, beta, size=13.0)
            chain(ax, 864, 1256, y, node_radius=7.1)
            centered(ax, 1321, y-28, q_label, size=17.0)
            arrow(ax, (1297, y-26), (1264, y-7), size=13, lw=1.25)
        rounded(ax, 696, 551, 653, 69, fill=LIGHT_BLUE)
        centered(ax, 1022, 585, "same network and damping support", size=12.2)

        # The two nudged states define the centered local parameter contrast.
        rounded(ax, 1414, 171, 170, 88, fill="#fff7f8", edge=RED,
                linewidth=1.25, radius=10)
        rounded(ax, 1414, 330, 170, 88, fill="#fbf7ff", edge=PURPLE,
                linewidth=1.25, radius=10)
        centered(ax, 1499, 215, r"$q_{+\beta}$", size=19)
        centered(ax, 1499, 374, r"$q_{-\beta}$", size=19)
        rounded(ax, 1633, 194, 379, 200, fill="#f7faff", edge="#435e84",
                linewidth=1.3, radius=12)
        arrow(ax, (1584, 215), (1628, 215), size=16, lw=1.55)
        arrow(ax, (1584, 374), (1628, 374), size=16, lw=1.55)
        centered(ax, 1822, 240, r"$g_{\beta,e}=$", size=17)
        centered(ax, 1823, 310,
                 r"$\frac{\partial_{\theta_e}E_e(q_{+\beta})-"
                 r"\partial_{\theta_e}E_e(q_{-\beta})}{2\beta}$", size=14.2)
        centered(ax, 1822, 422, "local centered estimator", size=12.4)
        rounded(ax, 1401, 551, 611, 80, fill=LIGHT_BLUE)
        centered(ax, 1706, 591,
                 "boundary flux records energy loss\n"
                 "phase contrast carries parameter credit", size=12.3)

        fig.savefig(png, dpi=160, bbox_inches=None, pad_inches=0)
        fig.savefig(pdf, bbox_inches=None, pad_inches=0)
        plt.close(fig)

    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("paper_figures"))
    args = parser.parse_args()
    png, pdf = render_architecture_figure(args.output_dir)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
