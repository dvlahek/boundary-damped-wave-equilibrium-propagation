#!/usr/bin/env python3
"""Presentation-only relabeling of the submitted BDW-EqProp figures.

The input images are the original *embedded image bytes* of Figure 5 and
Figure 10 in the 28-page revised main PDF, and of Figure S1 in the 5-page
revised supplementary PDF (both dated 2026-09-26).  The data-bearing graph
panels are never recomputed or re-digitized.  Only annotation-only rectangles
(three in-panel legends, two panel titles, one in-panel legend) are replaced;
a shared, larger legend is appended to the affected legend-heavy figures.

Usage:
  python figure_typography_revision.py --main-pdf revised_main.pdf \
      --supp-pdf revised_supplementary.pdf --output-dir submission_figures

Alternatively, supply --source-dir containing figure_5_original.png,
figure_10_original.png and figure_S1_original.png, extracted with this script.

Output filenames match the existing manuscript includegraphics references:
  figure_4.png   (main Figure 5)
  figure_mnist_training_audit.png  (main Figure 10)
  figure_5.png   (Supplementary Figure S1)
All three are additionally saved as print-ready PDFs. Original source PNGs and
SHA-256 provenance are retained under output-dir/source and manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

SOURCE_SPECS = {
    "figure_5_original.png": {"page": 15, "size": (2850, 855), "pdf": "main"},
    "figure_10_original.png": {"page": 22, "size": (2400, 1607), "pdf": "main"},
    "figure_S1_original.png": {"page": 4, "size": (2945, 855), "pdf": "supp"},
}
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
FONT_FILE = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def embedded_png(pdf_path: Path, page_number: int, dimensions: tuple[int, int]) -> bytes:
    with fitz.open(pdf_path) as pdf:
        p = pdf[page_number - 1]
        choices = []
        for entry in p.get_images(full=True):
            img = pdf.extract_image(entry[0])
            if (img["width"], img["height"]) == dimensions:
                choices.append(img["image"])
        if len(choices) != 1:
            raise ValueError(f"Expected exactly one embedded {dimensions} image on page "
                             f"{page_number} of {pdf_path}; found {len(choices)}")
        return choices[0]


def load_sources(args: argparse.Namespace, output: Path) -> dict[str, Image.Image]:
    sources = output / "source"
    sources.mkdir(parents=True, exist_ok=True)
    for name, spec in SOURCE_SPECS.items():
        dest = sources / name
        if args.source_dir:
            incoming = args.source_dir / name
            if not incoming.is_file():
                raise FileNotFoundError(incoming)
            dest.write_bytes(incoming.read_bytes())
        else:
            pdf = args.main_pdf if spec["pdf"] == "main" else args.supp_pdf
            if pdf is None:
                raise ValueError("Supply --main-pdf and --supp-pdf, or --source-dir")
            dest.write_bytes(embedded_png(pdf, spec["page"], spec["size"]))
        with Image.open(dest) as im:
            if im.size != spec["size"]:
                raise ValueError(f"Unexpected size for {name}: {im.size} vs {spec['size']}")
    return {name: Image.open(sources / name).convert("RGB") for name in SOURCE_SPECS}


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_FILE, size)


def erase_annotation(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int]) -> None:
    """Erase an annotation confined to graph background, not observed data marks."""
    draw.rectangle(rect, fill="white")


def restore_grid(img: Image.Image, rect: tuple[int, int, int, int], *,
                 row_y: int, col_x: int) -> None:
    """Continue visible gray background grid through a removed legend box.

    Samples are taken from adjacent untouched plot background. This reconstructs
    only axis grid lines where the original white legend already obscured them;
    no curves, points or bars occur in any of the three annotation rectangles.
    """
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = rect
    # Raw palette is RGB, and Matplotlib grid lines are neutral gray.
    def grid_color(color: tuple[int, ...]) -> bool:
        r, g, b = color[:3]
        return max(r, g, b) - min(r, g, b) < 3 and 195 <= r <= 252
    vertical = [(x, img.getpixel((x, row_y))) for x in range(x0, x1 + 1)]
    horizontal = [(y, img.getpixel((col_x, y))) for y in range(y0, y1 + 1)]
    erase_annotation(d, rect)
    for x, color in vertical:
        if grid_color(color):
            d.line((x, y0, x, y1), fill=color, width=1)
    for y, color in horizontal:
        if grid_color(color):
            d.line((x0, y, x1, y), fill=color, width=1)


def footer_legend(original: Image.Image, labels: tuple[str, ...], *,
                  font_size: int = 44, footer_height: int = 184) -> Image.Image:
    w, h = original.size
    canvas = Image.new("RGB", (w, h + footer_height), "white")
    canvas.paste(original, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.line((95, h + 8, w - 95, h + 8), fill="#b9c0c7", width=2)
    face = font(font_size)
    positions = ((200, h + 52), (w // 2 + 80, h + 52),
                 (200, h + 121), (w // 2 + 80, h + 121))
    for i, (label, (x, y)) in enumerate(zip(labels, positions)):
        color = COLORS[i]
        d.line((x, y + 22, x + 77, y + 22), fill=color, width=9)
        d.ellipse((x + 31, y + 13, x + 49, y + 31), fill=color)
        d.text((x + 100, y - 6), label, font=face, fill="#20262d")
    return canvas


def edit_figure_5(original: Image.Image) -> Image.Image:
    img = original.copy()
    d = ImageDraw.Draw(img)
    # Original legend boxes are in graph regions containing no data marks.
    # Their white legend backgrounds already occluded the grids at these sites.
    for rect, row, col in (((166, 602, 399, 728), 591, 415),
                           ((1095, 77, 1325, 197), 249, 1345),
                           ((2031, 77, 2262, 197), 247, 2278)):
        restore_grid(img, rect, row_y=row, col_x=col)
    return footer_legend(img, (
        "Boundary layer", "Endpoint only",
        "Distributed (local)", "Distributed (fixed trace)",
    ), font_size=46)


def edit_figure_10(original: Image.Image) -> Image.Image:
    img = original.copy()
    d = ImageDraw.Draw(img)
    heading = font(34)
    # Only heading whitespace is touched. Plot axes, lines, points and bars are unchanged.
    erase_annotation(d, (173, 936, 781, 970))
    erase_annotation(d, (1668, 936, 2373, 970))
    d.text((478, 952), "Finite-time gradient fidelity", font=heading,
           anchor="mm", fill="#20262d")
    d.text((2010, 952), "Final endpoint-gradient error", font=heading,
           anchor="mm", fill="#20262d")
    # Replace a development-only threshold label without touching the lines.
    d.rectangle((678, 1262, 787, 1285), fill="white")
    d.text((681, 1259), "1% target", font=font(20), fill="#20262d")
    return img


def edit_figure_S1(original: Image.Image) -> Image.Image:
    img = original.copy()
    d = ImageDraw.Draw(img)
    restore_grid(img, (668, 38, 952, 174), row_y=188, col_x=648)
    return footer_legend(img, (
        "Boundary dynamics", "Uniform dynamics",
        "Standard EqProp", "Implicit differentiation",
    ), font_size=49)


def write_outputs(output: Path, originals: dict[str, Image.Image]) -> dict:
    variants = (
        ("figure_5_original.png", "figure_4", edit_figure_5),
        ("figure_10_original.png", "figure_mnist_training_audit", edit_figure_10),
        ("figure_S1_original.png", "figure_5", edit_figure_S1),
    )
    provenance = {"method": "presentation-only image annotation; no data recomputation",
                  "source": "embedded PNGs from submitted 2026-09-26 PDFs",
                  "figures": {}}
    for source_name, stem, edit in variants:
        edited = edit(originals[source_name])
        png = output / (stem + ".png")
        pdf = output / (stem + ".pdf")
        edited.save(png, format="PNG", dpi=(300, 300), optimize=True)
        edited.save(pdf, format="PDF", resolution=300.0, title=stem)
        provenance["figures"][stem] = {
            "source_image": "source/" + source_name,
            "source_sha256": sha256(output / "source" / source_name),
            "png_sha256": sha256(png), "pdf_sha256": sha256(pdf),
            "output_size": list(edited.size),
        }
        print(stem, "OK", edited.size, sha256(png))
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n",
                                               encoding="utf-8")
    return provenance


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main-pdf", type=Path)
    p.add_argument("--supp-pdf", type=Path)
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("submission_figures"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = load_sources(args, args.output_dir)
    write_outputs(args.output_dir, sources)


if __name__ == "__main__":
    main()