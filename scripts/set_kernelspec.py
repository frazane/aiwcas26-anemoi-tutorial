#!/usr/bin/env python
"""Point notebooks at the tutorial's Jupyter kernel (registered by `source setup.sh`).

    python scripts/set_kernelspec.py                 # every notebook in the repository
    python scripts/set_kernelspec.py a.ipynb b.ipynb

Only metadata.kernelspec / language_info change; cells, outputs and attachments are kept as they
are (nothing is executed). Standard library only. Files are written in Jupyter's own format
(1-space indent, non-ASCII kept), so the git diff shows the metadata lines only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KERNELSPEC = {"display_name": "Python (aiwcas26 anemoi)", "language": "python", "name": "aiwcas26-anemoi"}
LANGUAGE_INFO = {
    "codemirror_mode": {"name": "ipython", "version": 3},
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
    "version": "3.11.7",
}


def dump(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]] or sorted(
        p for p in REPO.glob("**/*.ipynb") if "src" not in p.relative_to(REPO).parts[:1] and ".ipynb_checkpoints" not in p.parts
    )
    for path in paths:
        text = path.read_text()
        nb = json.loads(text)
        if dump(nb) != text:
            print(f"skip {path}: not in Jupyter's standard JSON layout, rewriting it would touch every line")
            continue
        nb["metadata"]["kernelspec"] = dict(KERNELSPEC)
        nb["metadata"]["language_info"] = {**LANGUAGE_INFO, **nb["metadata"].get("language_info", {})}
        new = dump(nb)
        if new != text:
            path.write_text(new)
            print(f"updated {path}")
        else:
            print(f"ok      {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
