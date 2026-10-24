"""Render notebooks/figures/*.svg to high-resolution PNGs next to them.

The PNGs are what scripts/embed_figures.py embeds in the notebooks
(GitHub does not render SVG cell attachments). Instructor-only; needs
cairosvg, e.g. in a throwaway venv: `pip install cairosvg`.

Usage: python scripts/render_figures.py [--width 2400] [svg ...]
"""

import argparse
from pathlib import Path

import cairosvg

FIGURES = Path(__file__).resolve().parent.parent / "notebooks" / "figures"

parser = argparse.ArgumentParser()
parser.add_argument("svgs", nargs="*", type=Path)
parser.add_argument("--width", type=int, default=2400, help="output width in pixels")
args = parser.parse_args()

for svg in args.svgs or sorted(FIGURES.glob("*.svg")):
    png = svg.with_suffix(".png")
    cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=args.width, background_color="white")
    print(f"{png.name}: {png.stat().st_size / 1e3:.0f} kB")
