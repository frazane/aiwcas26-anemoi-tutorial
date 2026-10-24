#!/usr/bin/env python
"""Check an anemoi-training config before you submit it (what `aiwcas validate` runs).

    python scripts/validate_config.py <config.yaml> [hydra overrides...] [--full]

Three checks, from the cheapest to the slowest:
  1. no `???` left in the file (unfilled TODOs), with the hint next to each one;
  2. the same schema check as `anemoi-training config validate` (Hydra compose + pydantic),
     with an absolute config path, so it works from any directory;
  3. a check that `config validate` does NOT do: every variable named in
     data.datasets.<name>.forcing / diagnostic must survive `select:` / `drop:` and exist in the
     dataset. Otherwise the job dies after startup with "Data indexing has invalid entries".
Exit code 0 when the config is fine.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

RED, GREEN, BOLD, END = ("\033[31m", "\033[32m", "\033[1m", "\033[0m") if sys.stdout.isatty() else ("",) * 4

# A few known error messages and what to do about them (see the table in exercises/README.md)
HINTS = [
    ("You must specify", "a `???` in the `defaults:` list: pick one of the options listed in the message"),
    ("Missing mandatory value", "a `???` is left in the file"),
    ("Extra inputs are not permitted", "a key the chosen component does not have (e.g. `num_heads` on a GNN): "
                                       "remove it, or use the right recipe block"),
    ("found duplicate key", "the same key appears twice in one section: replace the whole section instead of "
                            "pasting a second block under it"),
    ("precision", "training.precision must be a string: 32-true, 16-mixed or bf16-mixed"),
    ("Could not find", "a `defaults:` entry names a file that does not exist: check the spelling"),
]


def show(path: Path) -> str:
    """Path relative to the current directory when it is below it, else absolute."""
    rel = os.path.relpath(path)
    return str(path) if rel.startswith("..") else rel


def fail(msg: str) -> int:
    print(f"{RED}{BOLD}NOT OK{END}  {msg}")
    return 1


def unfilled_todos(path: Path) -> list[str]:
    """Lines with a ??? outside comments, each with its comment (or the TODO it belongs to) as a hint."""
    lines = path.read_text().splitlines()
    out = []
    for i, line in enumerate(lines):
        code, _, comment = line.partition("#")
        if "???" not in code:
            continue
        hint = comment.strip()
        if not hint:  # no comment on this line: point at the TODO(n) above it
            for j in range(i - 1, -1, -1):
                m = re.search(r"TODO\(\d+\)", lines[j])
                if m:
                    hint = f"see {m.group(0)} at line {j + 1}"
                    break
        out.append(f"  line {i + 1:4d}: {code.strip():32s} {('# ' + hint) if hint else ''}")
    return out


def _is_union_tag(part) -> bool:
    """pydantic labels the branches of a union: 'EnsModelSchema', 'function-after[...]', 'a.b.Class'."""
    p = str(part)
    return p.endswith("Schema") or p.startswith("function-after[") or "Schema]" in p or "." in p


def schema_errors(exc, full: bool) -> list[str]:
    """pydantic errors, shortened: for a union (e.g. the model types), only the best-matching branch."""
    groups: dict = {}
    for err in exc.errors():
        loc = err.get("loc", ())
        tags = tuple(str(p) for p in loc if _is_union_tag(p))
        groups.setdefault(str(loc[0]) if loc else "", {}).setdefault(tags, []).append(err)
    out, hidden = [], 0
    for branches in groups.values():
        keep = branches if full else {min(branches, key=lambda t: len(branches[t])): None}
        for tags in keep:
            for err in branches[tags]:
                path = ".".join(str(p) for p in err.get("loc", ()) if full or not _is_union_tag(p))
                extra = f" (you wrote {err.get('input')!r})" if err.get("type") == "extra_forbidden" else ""
                out.append(f"  {path}: {err.get('msg')}{extra}")
        hidden += sum(len(v) for v in branches.values()) - sum(len(branches[t]) for t in keep)
    if hidden:
        out.append(f"  ({hidden} similar messages for other model/training types hidden; add --full to see them)")
    return out


def check_variables(cfg: dict) -> list[str]:
    """forcing/diagnostic variables vs. select/drop and the dataset itself."""
    problems = []
    for name, dcfg in (cfg.get("data", {}).get("datasets") or {}).items():
        wanted = {kind: list(dcfg.get(kind) or []) for kind in ("forcing", "diagnostic")}
        n_before = len(problems)
        for kind, names in wanted.items():
            for v in names:
                not_selected, dropped = [], []
                for split in ("training", "validation", "test"):
                    ds = ((cfg.get("dataloader", {}).get(split) or {}).get("datasets") or {}).get(name) or {}
                    dsc = ds.get("dataset_config")
                    if not isinstance(dsc, dict):
                        continue
                    select = dsc.get("select")
                    select = [select] if isinstance(select, str) else select
                    if select is not None and v not in select:
                        not_selected.append(split)
                    if v in (dsc.get("drop") or []):
                        dropped.append(split)
                if not_selected:
                    problems.append(f"'{v}' is listed in data.datasets.{name}.{kind}, but `select:` "
                                    f"({'/'.join(not_selected)}) does not keep it. Add it to `select:`.")
                if dropped:
                    problems.append(f"'{v}' is listed in data.datasets.{name}.{kind}, but `drop:` "
                                    f"({'/'.join(dropped)}) removes it. Stop dropping it.")
        if len(problems) > n_before:
            continue
        # Names that are in no dataset at all (e.g. ERA5-only variables in the CERRA cutout of option B1)
        ds = ((cfg.get("dataloader", {}).get("training") or {}).get("datasets") or {}).get(name) or {}
        dsc = ds.get("dataset_config") or ds.get("dataset")
        if dsc is None:
            continue
        try:
            from anemoi.datasets import open_dataset

            if isinstance(dsc, dict):
                dsc = {k: v for k, v in dsc.items() if v is not None}
            variables = set(open_dataset(dsc).variables)
        except Exception as exc:  # noqa: BLE001
            first = str(exc).splitlines()[0][:300] if str(exc) else ""
            problems.append(f"could not open the training dataset (check the dataset paths): {type(exc).__name__}: {first}")
            continue
        for kind, names in wanted.items():
            missing = [v for v in names if v not in variables]
            if missing:
                problems.append(f"data.datasets.{name}.{kind} lists {', '.join(missing)}, which the dataset does not have. "
                                f"Write out the whole `{kind}:` list without them. The dataset has: "
                                f"{', '.join(sorted(variables))}")
    return problems


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0 if len(sys.argv) >= 2 else 2
    args = sys.argv[1:]
    full = "--full" in args
    args = [a for a in args if a != "--full"]
    path, overrides = Path(args[0]), args[1:]
    if not path.is_file():
        return fail(f"no such file: {path}")
    if path.suffix != ".yaml":
        return fail(f"{path} is not a .yaml file")
    path = path.resolve()
    shown = show(path)
    t0 = time.time()

    # 1. unfilled TODOs: instant, no imports
    todos = unfilled_todos(path)
    if todos:
        print(f"{RED}{BOLD}NOT OK{END}  {shown} still has {len(todos)} unfilled TODO(s) (???):")
        print("\n".join(todos))
        print("Replace every ??? with a value (the comment next to it says what), then run this again.")
        if path.name == "skeleton.yaml":
            mine = path.with_name("mine.yaml")
            print(f"Tip: do not edit the skeleton itself; work on a copy:\n  cp {shown} {show(mine)}")
        return 1

    # 2. schema check, as `anemoi-training config validate` does it
    print(f"Checking {shown} ...", flush=True)
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        from anemoi.training.schemas.base_schema import BaseSchema

        with initialize_config_dir(version_base=None, config_dir=str(path.parent)):
            cfg = compose(config_name=path.stem, overrides=overrides)
        BaseSchema(**cfg)
    except Exception as exc:  # noqa: BLE001
        text = str(exc).strip()
        print(f"{RED}{BOLD}NOT OK{END}  {shown}: {type(exc).__name__}")
        if hasattr(exc, "errors") and type(exc).__name__ == "ValidationError":
            print("\n".join(schema_errors(exc, full)))
        else:
            print("\n".join("  " + line for line in text.splitlines()[:25]))
        for key, hint in HINTS:
            if key in text:
                print(f"{BOLD}Hint:{END} {hint}")
                break
        return 1

    # 3. variables that `config validate` does not check
    problems = check_variables(OmegaConf.to_container(cfg, resolve=True))
    if problems:
        print(f"{RED}{BOLD}NOT OK{END}  {shown} passes the schema, but the job would fail after startup "
              "(\"Data indexing has invalid entries\"). `anemoi-training config validate` does not check this:")
        print("\n".join(f"  - {p}" for p in problems))
        return 1

    nxt = "" if os.environ.get("AIWCAS_SUBMITTING") else f" Submit it with:  aiwcas train {shown}"
    print(f"{GREEN}{BOLD}OK{END}  {shown} is valid ({time.time() - t0:.0f} s).{nxt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
