"""Embed figures referenced from markdown cells as notebook cell attachments.

GitHub's notebook viewer cannot resolve relative image paths, so figures
linked as ``figures/x.svg`` do not render there. This rewrites such links
to ``attachment:x.png`` and stores the PNG inside the cell, so the
notebook is self-contained. GitHub does not render SVG attachments, so
an SVG link embeds the PNG rendered next to it by render_figures.py.
Re-run both after editing a figure.

Usage: python scripts/embed_figures.py notebooks/03_models.ipynb [...]
"""

import base64
import json
import re
import sys
from pathlib import Path

MIME = {".png": "image/png"}
# ![alt](figures/x.svg) or <img src="figures/x.svg" alt="..." ...>
MD_RE = re.compile(r"!\[([^\]]*)\]\((figures/[^)\s]+)\)")
IMG_RE = re.compile(r'<img\s[^>]*src="(figures/[^"]+)"[^>]*>')
ALT_RE = re.compile(r'alt="([^"]*)"')


def embed(nb_path: Path) -> int:
    nb = json.loads(nb_path.read_text())
    n = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        src = "".join(cell["source"])
        attachments = cell.get("attachments", {})

        def attach(rel: str) -> str:
            path = nb_path.parent / rel
            if path.suffix == ".svg":
                path = path.with_suffix(".png")
            name = path.name
            data = base64.b64encode(path.read_bytes()).decode()
            attachments[name] = {MIME[path.suffix]: data}
            return name

        def md_sub(m):
            nonlocal n
            n += 1
            return f"![{m.group(1)}](attachment:{attach(m.group(2))})"

        def img_sub(m):
            nonlocal n
            n += 1
            alt = ALT_RE.search(m.group(0))
            return f"![{alt.group(1) if alt else ''}](attachment:{attach(m.group(1))})"

        new = IMG_RE.sub(img_sub, MD_RE.sub(md_sub, src))
        # Refresh attachments embedded by an earlier run from figures/.
        for name in list(attachments):
            if (nb_path.parent / "figures" / name).exists():
                attach(f"figures/{name}")
        if new != src or attachments:
            cell["source"] = new.splitlines(keepends=True)
            cell["attachments"] = attachments
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return n


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, embed(Path(p)), "figure(s) embedded")
