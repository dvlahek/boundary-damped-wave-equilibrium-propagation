#!/usr/bin/env python3
"""Presentation-only revision of manuscript Figures 3 and 6.

Figure 3 is edited only in four annotation text rectangles and a new large
footer. The original curves, markers, axes and grids are unmodified. Figure 6
is replotted from the locked modal-relaxation audit.csv, with the identical
median aggregations used in make_paper_figures.modal_rate_figure. The archived
numerical results are not recomputed or changed.

Example, using the original Figure 3 PNG shipped with the figure package:
  python figure_3_6_readability.py --figure3-source sources/figure_3_original.png \
    --modal-csv reported_results/modal_relaxation/audit.csv --output-dir out

Example, using the two locked plot-table exports in the figure package:
  python figure_3_6_readability.py --figure3-source sources/figure_3_original.png \
    --modal-summary-dir sources --output-dir out

For --main-pdf, provide the 28-page PDF with original Figure 3 on page 14.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

FONT_FILE = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")

def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_FILE, size)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def edit_figure_3(original: Image.Image) -> Image.Image:
    """Replace only legacy in-legend text, then add large keys below the panels.

    Existing legend boxes, color samples, axes, grids, curves, markers and
    numerical labels remain as drawn in the submitted manuscript source.
    """
    img = original.copy()
    d = ImageDraw.Draw(img)
    for x0, y0, x1, y1, text_y, label in (
        (1174, 655, 1353, 688, 655, "Boundary layer"),
        (1174, 690, 1353, 722, 688, "Endpoint only"),
        (2631, 86, 2809, 119, 86, "Boundary layer"),
        (2631, 120, 2809, 152, 118, "Endpoint only"),
    ):
        d.rectangle((x0, y0, x1, y1), fill="white")
        d.text((x0 + 2, text_y), label, font=font(22), fill="#20262d")
    w, h = img.size
    canvas = Image.new("RGB", (w, h + 226), "white")
    canvas.paste(img)
    d = ImageDraw.Draw(canvas)
    d.line((115, h + 9, w - 115, h + 9), fill="#b9c0c7", width=2)
    d.text((155, h + 26), "Energy balance (left)", font=font(45), fill="#20262d")
    d.text((1490, h + 26), "Damping support (middle and right)",
           font=font(45), fill="#20262d")
    for x, yy, label, color in (
        (155, h + 99, "Energy above equilibrium", COLORS[0]),
        (155, h + 165, "Energy-balance residual", COLORS[1]),
        (1490, h + 99, "Boundary layer", COLORS[0]),
        (1490, h + 165, "Endpoint only", COLORS[1]),
    ):
        d.line((x, yy + 18, x + 77, yy + 18), fill=color, width=9)
        d.ellipse((x + 32, yy + 10, x + 49, yy + 27), fill=color)
        d.text((x + 101, yy - 10), label, font=font(44), fill="#20262d")
    return canvas


def _median_modal_tables(full_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exactly reproduce the aggregations of make_paper_figures.modal_rate_figure.

    Each plotted point is a median over fixed rows of the archived CSV.  No
    physical solver and no new numerical experiment is executed here.
    """
    data = pd.read_csv(full_csv)
    required = {"seed", "chain_size", "variant", "eta", "exact_decay_rate",
                "first_order_predicted_decay_rate", "first_order_decay_coefficient"}
    if not required <= set(data):
        raise ValueError(f"Missing modal source columns: {sorted(required - set(data))}")
    left = data.groupby(["variant", "eta"], as_index=False).agg(
        exact=("exact_decay_rate", "median"),
        predicted=("first_order_predicted_decay_rate", "median"),
    )
    small = data.loc[data.groupby(["seed", "chain_size", "variant"])["eta"].idxmin()]
    right = small.groupby(["variant", "chain_size"], as_index=False).agg(
        coeff=("first_order_decay_coefficient", "median"))
    return left, right


def _modal_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    source = Path(args.modal_csv) if args.modal_csv else None
    if source is not None:
        left, right = _median_modal_tables(source)
        return left, right, {"source": str(source), "source_sha256": sha256(source),
                             "aggregation": "fixed medians from locked archived rows"}
    root = args.modal_summary_dir if args.modal_summary_dir else args.output_dir / "sources"
    left_path = root / "modal_plot_data.csv"
    right_path = root / "modal_participation_data.csv"
    if not (left_path.exists() and right_path.exists()):
        raise ValueError("Supply --modal-csv from reported_results or --source-dir "
                         "containing both modal plot tables")
    return pd.read_csv(left_path), pd.read_csv(right_path), {
        "source": "modal_plot_data.csv + modal_participation_data.csv",
        "modal_plot_sha256": sha256(left_path),
        "modal_participation_sha256": sha256(right_path),
        "source_repo_blob_sha": "86ffa1d700f951be6a3efe5899d658c5b6b5140c",
        "aggregation": "locked v1.0.0 CSV medians, no physical re-simulation"}


def render_figure_6(output: Path, left: pd.DataFrame, right: pd.DataFrame) -> tuple[Path, Path]:
    """Large-font modal plot from archived values; consistent color across panels."""
    cmap = {"paper_boundary": ("Boundary layer", "#1f77b4"),
            "single_terminal": ("Endpoint only", "#ff7f0e"),
            "uniform_trace": ("Distributed (fixed trace)", "#2ca02c")}
    variants = ("paper_boundary", "single_terminal", "uniform_trace")
    if set(left["variant"]) != set(variants) or set(right["variant"]) != set(variants):
        raise ValueError("Modal tables do not contain precisely the three reported supports")
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.8))
    fig.subplots_adjust(left=0.082, right=0.985, top=0.89, bottom=0.285, wspace=0.34)
    for variant in variants:
        label, color = cmap[variant]
        a = left.loc[left["variant"] == variant].sort_values("eta")
        b = right.loc[right["variant"] == variant].sort_values("chain_size")
        axes[0].loglog(a["eta"], a["predicted"], color=color, ls="--",
                       lw=2.2, alpha=0.96, zorder=2)
        axes[0].loglog(a["eta"], a["exact"], color=color, ls="-", lw=1.5,
                       marker="o", ms=8.2, zorder=3)
        axes[1].loglog(b["chain_size"], b["coeff"], color=color,
                       lw=2.25, marker="o", ms=8.2)
    axes[0].set(title="Weak-damping modal law", xlabel=r"Damping scale $\eta$",
                ylabel="Slowest decay rate")
    axes[1].set(title="Boundary participation bottleneck", xlabel="Chain size",
                ylabel="Slowest modal participation coefficient")
    axes[1].xaxis.set_major_locator(FixedLocator([3, 5, 9, 17, 33]))
    axes[1].xaxis.set_major_formatter(ScalarFormatter())
    for ax in axes:
        ax.grid(alpha=0.23, which="both")
        ax.tick_params(axis="both", which="major", labelsize=16, pad=7)
        ax.tick_params(axis="both", which="minor", labelsize=13)
        ax.xaxis.label.set_fontsize(17)
        ax.yaxis.label.set_fontsize(17)
        ax.set_title(ax.get_title(), fontsize=20, pad=17)
    handles = [
        Line2D([0], [0], color=cmap["paper_boundary"][1], marker="o", lw=2.3, ms=8),
        Line2D([0], [0], color="black", marker="o", lw=1.5, ms=7.5),
        Line2D([0], [0], color=cmap["single_terminal"][1], marker="o", lw=2.3, ms=8),
        Line2D([0], [0], color="black", ls="--", lw=2.2),
        Line2D([0], [0], color=cmap["uniform_trace"][1], marker="o", lw=2.3, ms=8),
    ]
    labels = ["Boundary layer", "Exact", "Endpoint only", "First-order prediction",
              "Distributed (fixed trace)"]
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.026), fontsize=16.5, frameon=False,
               columnspacing=2.4, handlelength=2.5, handletextpad=0.8)
    png = output / "figure_modal_relaxation.png"
    pdf = output / "figure_modal_relaxation.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white", metadata={"Title": "Weak-damping modal relaxation"})
    plt.close(fig)
    return png, pdf



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figure3-source", type=Path)
    ap.add_argument("--main-pdf", type=Path)
    ap.add_argument("--modal-csv", type=Path)
    ap.add_argument("--modal-summary-dir", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.figure3_source:
        raw = args.figure3_source.read_bytes()
    elif args.main_pdf:
        with fitz.open(args.main_pdf) as doc:
            matches = [doc.extract_image(x[0])["image"] for x in doc[13].get_images(full=True)
                       if x[2:4] == (2850, 855)]
        if len(matches) != 1:
            raise ValueError("Expected one original 2850x855 Figure 3 image on page 14")
        raw = matches[0]
    else:
        raise ValueError("Supply either --figure3-source or --main-pdf")
    source = args.output_dir / "figure_3_original.png"
    source.write_bytes(raw)
    img = Image.open(source).convert("RGB")
    if img.size != (2850, 855):
        raise ValueError("Figure 3 source size mismatch: expected 2850x855")
    edited = edit_figure_3(img)
    f3png = args.output_dir / "figure_2.png"
    f3pdf = args.output_dir / "figure_2.pdf"
    edited.save(f3png, format="PNG", dpi=(300, 300), optimize=True)
    edited.save(f3pdf, format="PDF", resolution=300.0, title="figure_2")
    if not args.modal_csv and not args.modal_summary_dir:
        default = Path("reported_results/modal_relaxation/audit.csv")
        if default.is_file():
            args.modal_csv = default
    if args.modal_summary_dir:
        args.modal_summary_dir = args.modal_summary_dir.resolve()
    left, right, origin = _modal_tables(args)
    f6png, f6pdf = render_figure_6(args.output_dir, left, right)
    manifest = {"script_sha256": sha256(Path(__file__)), "figures": {
        "figure_2": {"original_sha256": sha256(source),
                     "png_sha256": sha256(f3png), "pdf_sha256": sha256(f3pdf),
                     "description": "annotation-only revision of original embedded PNG"},
        "figure_modal_relaxation": {**origin, "png_sha256": sha256(f6png),
                                    "pdf_sha256": sha256(f6pdf),
                                    "description": "locked modal medians, enlarged scientific typography"}
    }}
    (args.output_dir / "figure_3_6_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, detail in manifest["figures"].items():
        print(name, "PASS", detail["png_sha256"])

if __name__ == "__main__":
    main()
