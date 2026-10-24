"""Generate notebooks/figures/02_encoder_processor_decoder.svg (static, hand-tuned layout)."""
import math
import sys

out = sys.argv[1]

W, H = 1000, 470
BLUE = "#2a78d6"
BLUE_L = "#dbeafe"
ORANGE = "#eb6834"
ORANGE_L = "#ffedd5"
INK = "#1f2937"
MUTED = "#6b7280"
EDGE = "#9ca3af"

parts = []
add = parts.append

add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    'font-family="Helvetica, Arial, sans-serif">')
add("""<defs>
  <marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#374151"/>
  </marker>
  <marker id="aho" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#c2410c"/>
  </marker>
</defs>""")
add(f'<rect x="0" y="0" width="{W}" height="{H}" rx="14" fill="#ffffff"/>')

# panel geometry
PY0, PY1 = 92, 332
panels = {"A": (30, 270), "B": (380, 620), "C": (730, 970)}


def panel(x0, x1, fill, stroke):
    add(f'<rect x="{x0}" y="{PY0}" width="{x1 - x0}" height="{PY1 - PY0}" rx="12" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.2"/>')


def data_grid(x0, x1, r=2.3):
    """Reduced-Gaussian-like grid drawn in lat-lon: fewer points per row towards the poles."""
    nrows = 14
    pad = 14
    for i in range(nrows):
        lat = 90 - (i + 0.5) * 180 / nrows
        y = PY0 + pad + (i + 0.5) * (PY1 - PY0 - 2 * pad) / nrows
        npts = max(4, round(30 * math.cos(math.radians(lat))))
        for j in range(npts):
            x = x0 + pad + (j + 0.5) * (x1 - x0 - 2 * pad) / npts
            add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{BLUE}"/>')


# ---- panel A and C: data nodes
panel(*panels["A"], "#f8fafc", "#cbd5e1")
data_grid(*panels["A"])
panel(*panels["C"], "#f8fafc", "#cbd5e1")
data_grid(*panels["C"])

# ---- panel B: hidden mesh (triangular lattice, two scales)
bx0, bx1 = panels["B"]
panel(bx0, bx1, "#fffaf5", "#fed7aa")
cols, rows = 7, 7
dx = (bx1 - bx0 - 40) / (cols - 1 + 0.5)
dy = (PY1 - PY0 - 40) / (rows - 1)
nodes = {}
for r in range(rows):
    for c in range(cols):
        x = bx0 + 20 + c * dx + (dx / 2 if r % 2 else 0)
        y = PY0 + 20 + r * dy
        nodes[(r, c)] = (x, y)


def neighbours(r, c, step=1):
    """Right, down-left and down-right neighbours in an offset triangular lattice."""
    out = [(r, c + step)]
    if step == 1:
        off = c if r % 2 == 0 else c + 1
        out += [(r + 1, off - 1), (r + 1, off)]
    return out


# fine edges
for (r, c), (x, y) in nodes.items():
    for nb in neighbours(r, c):
        if nb in nodes:
            x2, y2 = nodes[nb]
            add(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{EDGE}" stroke-width="1"/>')

# coarse edges (multi-scale): every 2nd row, every 2nd column -> again a triangular lattice
coarse = {(r, c): nodes[(r, c)] for (r, c) in nodes if (r % 4 == 0 and c % 2 == 0) or (r % 4 == 2 and c % 2 == 1)}
for (r, c), (x, y) in coarse.items():
    # coarse lattice = every 2nd node of every 2nd row; neighbours: right, down-left, down-right
    for nb in [(r, c + 2), (r + 2, c - 1), (r + 2, c + 1)]:
        if nb in coarse:
            x2, y2 = coarse[nb]
            add(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{ORANGE}" '
                'stroke-opacity="0.6" stroke-width="2.6" stroke-linecap="round"/>')

for (r, c), (x, y) in nodes.items():
    big = (r, c) in coarse
    add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{6 if big else 4.2}" fill="{ORANGE}" stroke="#ffffff" stroke-width="1.5"/>')

# ---- stage labels above panels
def label(x, y, text, size=17, weight="700", color=INK, anchor="middle"):
    add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{text}</text>')


label(150, 56, "Data nodes", color=BLUE)
label(150, 76, "input state  x(t)", size=13, weight="400", color=MUTED)
label(500, 56, "Hidden nodes", color=ORANGE)
label(500, 76, "latent mesh", size=13, weight="400", color=MUTED)
label(850, 56, "Data nodes", color=BLUE)
label(850, 76, "forecast  x(t + Δt)", size=13, weight="400", color=MUTED)

# ---- arrows between panels
def arrow(x0, x1, y, title, sub1, sub2):
    add(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#374151" stroke-width="2.2" marker-end="url(#ah)"/>')
    xm = (x0 + x1) / 2
    label(xm, y - 14, title, size=15)
    label(xm, y + 22, sub1, size=11.5, weight="400", color=MUTED)
    label(xm, y + 37, sub2, size=11.5, weight="400", color=MUTED)


arrow(278, 372, 212, "Encoder", "data → hidden", "CutOffEdges")
arrow(628, 722, 212, "Decoder", "hidden → data", "KNNEdges (k = 3)")

# processor loop below panel B
add(f'<path d="M560,{PY1 + 8} C600,{PY1 + 48} 400,{PY1 + 48} 440,{PY1 + 10}" fill="none" stroke="#c2410c" '
    'stroke-width="2.2" marker-end="url(#aho)"/>')
label(500, PY1 + 60, "Processor", size=15, color="#c2410c")
label(500, PY1 + 77, "hidden → hidden, repeated message passing", size=11.5, weight="400", color=MUTED)
label(500, PY1 + 92, "MultiScaleEdges: fine (grey) + coarse (orange) links", size=11.5, weight="400", color=MUTED)

# captions under data panels
for xc, l1 in [(150, "the dataset grid, e.g. O96"), (850, "same grid as the input")]:
    label(xc, PY1 + 24, "one node per gridpoint", size=11.5, weight="400", color=MUTED)
    label(xc, PY1 + 39, l1, size=11.5, weight="400", color=MUTED)

# bottom band: physical / latent / physical
yb = H - 24
for xc, t, col in [(150, "physical space", BLUE), (500, "latent space", ORANGE), (850, "physical space", BLUE)]:
    label(xc, yb, t, size=12, weight="600", color=col)
add(f'<line x1="30" y1="{yb - 18}" x2="970" y2="{yb - 18}" stroke="#e5e7eb" stroke-width="1"/>')

add("</svg>")
with open(out, "w") as f:
    f.write("\n".join(parts) + "\n")
print("wrote", out)
