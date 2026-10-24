#!/usr/bin/env python
"""Map of the option-B1 limited-area graph: CERRA points over Italy, the ERA5 boundary ring and the
hidden mesh, plus a global inset showing where the LAM sits inside ERA5 O96.

    python scripts/plot_lam_graph.py                         # the solution's cached graph
    python scripts/plot_lam_graph.py --graph $SCRATCH/aiwcas26/hour2/graphs/lam_italy_thin4_tri8.pt
    python scripts/plot_lam_graph.py --out exercises/B1_lam/lam_graph   # writes .png and .svg

The graph file is written by the first training run (or by `scripts/check_exercises.py
exercises/B1_lam/solution.yaml --cpu-steps 2`). Needs matplotlib + cartopy (in the tutorial env);
the Natural Earth coastlines come from $WORK/a08trd25/data/cartopy (scripts/download_data.sh).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
DATA = Path("/leonardo_work/tra26_aiwcas/a08trd25/data")
GRAPH = Path(os.environ.get("SCRATCH", ".")) / "aiwcas26/hour2/graphs/lam_italy_thin4_tri7.pt"
O96 = DATA / "anemoi/aifs-ea-an-oper-0001-mars-o96-2017-2017-6h-v8-testing.zarr"

# Same reference palette as exercises/compare.ipynb
LAM, BOUND, HIDDEN = "#2a78d6", "#eb6834", "#0b0b0b"
INK, INK2, GRID, SURFACE, LAND = "#0b0b0b", "#52514e", "#c9c8c3", "#fcfcfb", "#f0efeb"


def latlon(nodes) -> tuple[np.ndarray, np.ndarray]:
    x = np.degrees(nodes.x.numpy())
    lon = np.where(x[:, 1] > 180, x[:, 1] - 360, x[:, 1])
    return x[:, 0], lon


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default=str(GRAPH))
    ap.add_argument("--out", default=str(REPO / "exercises/B1_lam/lam_graph"), help="output path without extension")
    args = ap.parse_args()

    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    cartopy.config["data_dir"] = str(DATA / "cartopy")

    g = torch.load(args.graph, weights_only=False, map_location="cpu")
    dlat, dlon = latlon(g["data"])
    lam = g["data"]["cutout_mask"].squeeze().numpy().astype(bool)
    hlat, hlon = latlon(g["hidden"])
    hh = g["hidden", "to", "hidden"].edge_index.numpy()
    print(f"data nodes {lam.size} (LAM {lam.sum()}, boundary {(~lam).sum()}), hidden {hlat.size}, "
          f"hidden edges {hh.shape[1]}")

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.titlecolor": INK,
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig = plt.figure(figsize=(11, 6.6), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1])
    proj = ccrs.LambertConformal(central_longitude=12.5, central_latitude=42, standard_parallels=(38, 46))
    ax = fig.add_subplot(gs[0], projection=proj)
    ax.set_extent([-1.5, 26.5, 31.0, 51.5], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND, edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=INK2, linewidth=0.6, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=GRID, linewidth=0.5, zorder=1)
    pc = ccrs.PlateCarree()

    # hidden -> hidden edges (great circles are near-straight at this scale)
    seg = np.stack([np.c_[hlon[hh[0]], hlat[hh[0]]], np.c_[hlon[hh[1]], hlat[hh[1]]]], axis=1)
    seg_xy = proj.transform_points(pc, seg[..., 0].ravel(), seg[..., 1].ravel())[:, :2].reshape(seg.shape)
    ax.add_collection(LineCollection(seg_xy, colors=HIDDEN, linewidths=0.35, alpha=0.35, zorder=3))
    ax.scatter(dlon[lam], dlat[lam], s=1.2, c=LAM, lw=0, transform=pc, zorder=2, rasterized=True)
    ax.scatter(dlon[~lam], dlat[~lam], s=26, c=BOUND, edgecolors=SURFACE, linewidths=0.8, transform=pc, zorder=4)
    ax.scatter(hlon, hlat, s=9, c=HIDDEN, edgecolors=SURFACE, linewidths=0.5, transform=pc, zorder=5)
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, color=GRID, linewidth=0.4, linestyle=":")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"color": INK2, "size": 8}
    ax.set_title("Option B1 graph: what the model sees", loc="left", fontsize=12, fontweight="bold")

    # direct labels
    lab = {"fontsize": 9, "transform": pc, "zorder": 6,
           "bbox": {"facecolor": SURFACE, "edgecolor": GRID, "linewidth": 0.6, "alpha": 0.95, "boxstyle": "round,pad=0.3"}}
    ax.text(14.6, 47.6, f"CERRA (LAM)\n{lam.sum()} points, 22 km", color=INK, ha="left", **lab)
    ax.text(-0.8, 50.3, f"ERA5 O96 boundary\n{(~lam).sum()} points within 300 km", color=INK, ha="left", **lab)
    ax.text(19.6, 31.6, f"hidden mesh\n{hlat.size} nodes (~60 km)", color=INK, ha="left", **lab)

    handles = [Line2D([], [], ls="", marker="o", ms=4, mfc=LAM, mec="none", label="CERRA points (predicted, in the loss)"),
               Line2D([], [], ls="", marker="o", ms=6, mfc=BOUND, mec=SURFACE, label="ERA5 points (boundary forcing only)"),
               Line2D([], [], ls="-", marker="o", ms=4, color=HIDDEN, lw=0.6, mec=SURFACE, label="hidden nodes and edges (processor)")]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, frameon=True, facecolor=SURFACE, edgecolor=GRID,
              framealpha=0.95)

    # global inset: the whole O96 grid with the LAM in it
    gax = fig.add_subplot(gs[1], projection=ccrs.Orthographic(central_longitude=12, central_latitude=35))
    gax.set_global()
    gax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=LAND, edgecolor="none", zorder=0)
    gax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=INK2, linewidth=0.4, zorder=1)
    try:
        import zarr

        z = zarr.open(str(O96), mode="r")
        olat, olon = z["latitudes"][:], z["longitudes"][:]
        gax.scatter(olon, olat, s=0.6, c=GRID, lw=0, transform=pc, zorder=1, rasterized=True)
    except Exception as exc:  # noqa: BLE001
        print(f"(no O96 background: {exc})")
    gax.scatter(dlon[lam], dlat[lam], s=0.5, c=LAM, lw=0, transform=pc, zorder=2, rasterized=True)
    gax.scatter(dlon[~lam], dlat[~lam], s=3, c=BOUND, lw=0, transform=pc, zorder=3)
    gax.set_title("ERA5 O96 (40,320 points): only\nthe ring around Italy is kept", loc="left", fontsize=10)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=160 if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {out.with_suffix('.' + ext)}")


if __name__ == "__main__":
    main()
