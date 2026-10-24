#!/usr/bin/env python
"""Plot the loss curves of one of YOUR runs (reads the offline MLflow files; no server needed).

    python scripts/plot_loss.py --job 12345678          # the Slurm job id printed by submit.sh
    python scripts/plot_loss.py --run-dir <.../logs/mlflow/<exp>/<run_id>>
    python scripts/plot_loss.py --latest                 # your most recent hour-2 run

Prints first -> last losses and writes a PNG next to the Slurm log
($SCRATCH/aiwcas26/logs/loss-<job>.png); open it in JupyterLab's file browser.
The room-wide comparison is exercises/compare.ipynb.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoreboard import find_run_dir, pick_metric, read_metric  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--job")
    g.add_argument("--run-dir")
    g.add_argument("--latest", action="store_true")
    ap.add_argument("-o", "--output", help="PNG path")
    args = ap.parse_args()

    root = os.path.join(os.environ.get("SCRATCH", "."), "aiwcas26")
    if args.run_dir:
        run = Path(args.run_dir)
    elif args.job:
        run = find_run_dir(args.job, [root])
    else:
        runs = [Path(p).parent.parent for p in glob.glob(os.path.join(root, "hour2", "*", "logs", "mlflow", "*", "*", "metrics"))]
        run = max(runs, key=lambda p: p.stat().st_mtime) if runs else None
    if run is None or not run.is_dir():
        print("no run found (is the job finished? did it reach the first logged step?)", file=sys.stderr)
        return 1

    name = (run / "tags" / "mlflow.runName").read_text().strip() if (run / "tags" / "mlflow.runName").is_file() else run.name
    tr = read_metric(pick_metric(run / "metrics", "train_multi_dataset_loss_step", "train_*loss_step") or Path("-"))
    va = read_metric(pick_metric(run / "metrics", "val_multi_dataset_loss_epoch", "val_*loss_epoch") or Path("-"))
    if not tr:
        print(f"{run}: no training loss logged yet", file=sys.stderr)
        return 1
    print(f"run {name}  ({run})")
    print(f"train loss: {tr[0][1]:.4g} (step {tr[0][2] + 1}) -> {tr[-1][1]:.4g} (step {tr[-1][2] + 1})")
    if va:
        print(f"val loss  : {va[0][1]:.4g} (step {va[0][2] + 1}) -> {va[-1][1]:.4g} (step {va[-1][2] + 1})")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
    ax.plot([s + 1 for _, _, s in tr], [v for _, v, _ in tr], color="#2a6fb0", lw=1.2, label="train (per step)")
    if va:
        ax.plot([s + 1 for _, _, s in va], [v for _, v, _ in va], "o-", color="#d9822b", ms=3, lw=1.5, label="validation (per epoch)")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title(name, loc="left", fontsize=11)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    tag = args.job or run.name[:8]
    out = Path(args.output or os.path.join(root, "logs", f"loss-{tag}.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"plot: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
