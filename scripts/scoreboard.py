#!/usr/bin/env python
"""Hour-2 scoreboard: one CSV row per finished training run, shared by the whole room.

Standard library only (no anemoi/torch import), so it runs in a second at the end of a job.

    python scripts/scoreboard.py add --job 12345678     # what scripts/train.sbatch runs for you
    python scripts/scoreboard.py add --run-dir <.../logs/mlflow/<exp>/<run_id>>   # by hand
    python scripts/scoreboard.py show                    # print the room's table
    python scripts/scoreboard.py init                    # instructor, once: create the shared dir

`add` finds the offline-MLflow run of the job (tag SLURM_JOB_ID) under $SCRATCH/aiwcas26, then
  * appends one row to   $SCOREBOARD_DIR/scoreboard.csv
  * copies the loss curves (a few kB) to $SCOREBOARD_DIR/curves/<user>_<job>/ because
    everybody's $SCRATCH is private (drwx------) and the compare notebook must read them.
$SCOREBOARD_DIR defaults to /leonardo_work/tra26_aiwcas/aiwcas26_scoreboard (group tra26_aiwcas).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import glob
import hashlib
import io
import json
import os
import re
import shutil
import statistics
import sys
from pathlib import Path

DEFAULT_DIR = "/leonardo_work/tra26_aiwcas/aiwcas26_scoreboard"
GROUP = "tra26_aiwcas"

COLUMNS = [
    "finished_at", "user", "option", "run_name", "job_id", "mlflow_run_id",
    "arch", "loss", "key_params", "compare_group",
    "n_params", "n_out_vars", "steps", "s_per_step", "peak_gpu_mem_mib",
    "train_loss_first", "train_loss_last", "val_loss_first", "val_loss_last", "loss_ratio",
    "startup_s", "train_cmd_wall_s", "curves_dir", "config_file",
]


# ----------------------------------------------------------------------------- helpers
def scoreboard_dir(arg: str | None) -> Path:
    return Path(arg or os.environ.get("SCOREBOARD_DIR") or DEFAULT_DIR)


def read_metric(path: Path) -> list[tuple[int, float, int]]:
    """MLflow file-store metric: one 'timestamp_ms value step' per line."""
    rows = []
    if path.is_file():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                rows.append((int(parts[0]), float(parts[1]), int(parts[2])))
    return rows


def pick_metric(metrics_dir: Path, preferred: str, pattern: str) -> Path | None:
    p = metrics_dir / preferred
    if p.is_file():
        return p
    hits = sorted(metrics_dir.glob(pattern))
    return hits[0] if hits else None


def find_run_dir(job: str, roots: list[str]) -> Path | None:
    hits = []
    for root in roots:
        for tag in glob.glob(os.path.join(root, "**", "mlflow", "*", "*", "tags", "SLURM_JOB_ID"), recursive=True):
            try:
                if Path(tag).read_text().strip() == str(job):
                    hits.append(Path(tag).parent.parent)
            except OSError:
                pass
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def parse_n_params(log: Path | None) -> str:
    """Lightning's model summary line, e.g. 'Total params: 2.3 M' -> '2300000'."""
    if not log or not log.is_file():
        return ""
    m = None
    with log.open(errors="replace") as fh:
        for line in fh:
            if "Total params" in line:
                m = re.search(r"Total params:\s*([\d.]+)\s*([KMBT]?)", line)
    if not m:
        return ""
    scale = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[m.group(2)]
    return str(int(float(m.group(1)) * scale))


def peak_gpu_mem(gpu_csv: Path | None) -> str:
    if not gpu_csv or not gpu_csv.is_file():
        return ""
    vals = [int(x) for x in re.findall(r"(\d+)\s*(?:MiB)?\s*$", gpu_csv.read_text(), flags=re.M)]
    return str(max(vals)) if vals else ""


def short_target(target: str | None) -> str:
    return (target or "?").rsplit(".", 1)[-1]


def describe(cfg: dict, metadata: dict) -> dict:
    """Pull the knobs that matter for the comparison out of the logged config."""
    model, task, training = cfg.get("model", {}), cfg.get("task", {}), cfg.get("training", {})
    arch = short_target(model.get("processor", {}).get("_target_")).replace("Processor", "")
    method = short_target(training.get("method", {}).get("_target_"))
    ens = training.get("ensemble_size_per_device") if method == "EnsembleTraining" else None

    loss_cfg = training.get("training_loss", {}).get("datasets", {}).get("data", {})
    loss = short_target(loss_cfg.get("_target_"))
    if loss == "MultiscaleLossWrapper":
        inner = loss_cfg.get("per_scale_loss", {})
        loss = short_target(inner.get("_target_"))
        if loss == "CRPS":
            alpha = inner.get("alpha")
            loss = "afCRPS" if alpha not in (None, 1.0, 0.0) else ("fCRPS" if alpha == 1.0 else "CRPS")
            loss += f"(a={alpha})"
    loss = loss.replace("Loss", "")

    rollout = task.get("rollout", {})
    r_start, r_max = rollout.get("start"), rollout.get("maximum")
    roll = f"{r_start}" if r_start == r_max else f"{r_start}..{r_max}"
    try:
        n_out = len(metadata["data_indices"]["data"]["model"]["output"]["full"])
    except (KeyError, TypeError):
        n_out = ""
    graph_file = Path(str(cfg.get("system", {}).get("input", {}).get("graph", ""))).stem

    key = [f"arch={arch}", f"ch={model.get('num_channels') or model.get('processor', {}).get('num_channels')}", f"L={model.get('processor', {}).get('num_layers')}",
           f"ms={task.get('multistep_input')}", f"roll={roll}", f"vars={n_out}",
           f"lr={training.get('optimization', {}).get('lr')}", f"steps={training.get('max_steps')}"]
    if ens:
        key.insert(1, f"ens={ens}")
    if graph_file:
        key.append(f"graph={graph_file}")
    # Raw losses are only comparable with the same loss, outputs, rollout and scaling.
    weights = training.get("scalers", {}).get("datasets", {}).get("data", {}).get("general_variable", {}).get("weights", {})
    # a short fingerprint of the loss weights (the full dict is too long for a plot title)
    wtag = "weights#" + hashlib.sha1(json.dumps(weights, sort_keys=True).encode()).hexdigest()[:6] if weights else ""
    group = f"{loss} | vars={n_out} | roll={roll}"
    return {
        "arch": arch, "loss": loss, "key_params": " ".join(key), "n_out_vars": n_out,
        "compare_group": group, "_wtag": wtag,
    }


def fmt(x: float | None, nd: int = 5) -> str:
    return "" if x is None else f"{x:.{nd}g}"


# ----------------------------------------------------------------------------- add
def build_row(run_dir: Path, args: argparse.Namespace) -> tuple[dict, dict]:
    tags = {p.name: p.read_text().strip() for p in (run_dir / "tags").glob("*") if p.is_file()}
    meta_yaml = (run_dir / "meta.yaml").read_text() if (run_dir / "meta.yaml").is_file() else ""
    exp_dir = run_dir.parent
    exp_meta = (exp_dir / "meta.yaml").read_text() if (exp_dir / "meta.yaml").is_file() else ""
    m = re.search(r"^name:\s*(.+)$", exp_meta, flags=re.M)
    experiment = m.group(1).strip().strip("'\"") if m else ""
    option = os.environ.get("EXERCISE_OPTION") or (experiment.split("hour2-", 1)[1] if "hour2-" in experiment else experiment)

    cfg, metadata = {}, {}
    cfg_json = run_dir / "artifacts" / "config.json"
    if cfg_json.is_file():
        blob = json.loads(cfg_json.read_text())
        metadata = blob.get("metadata", {}) or {}
        cfg = metadata.get("config", {}) or {}
    info = describe(cfg, metadata)
    if info["_wtag"] and option == "0":
        info["compare_group"] += f" | {info['_wtag']}"

    metrics = run_dir / "metrics"
    tr_path = pick_metric(metrics, "train_multi_dataset_loss_step", "train_*loss_step")
    va_path = pick_metric(metrics, "val_multi_dataset_loss_epoch", "val_*loss_epoch")
    train, val = read_metric(tr_path) if tr_path else [], read_metric(va_path) if va_path else []

    def first(xs, k=3):
        return statistics.fmean(v for _, v, _ in xs[:k]) if xs else None

    def last(xs, k=3):
        return statistics.fmean(v for _, v, _ in xs[-k:]) if xs else None

    s_per_step = None
    if len(train) >= 2 and train[-1][2] > train[0][2]:
        s_per_step = (train[-1][0] - train[0][0]) / 1000 / (train[-1][2] - train[0][2])
    startup = None
    if train and args.train_start:
        # time from `anemoi-training train` to the first logged loss: imports, setup, graph, first steps
        startup = train[0][0] / 1000 - args.train_start
    t_first, t_last = first(train), last(train)

    user = tags.get("mlflow.user") or os.environ.get("USER", "?")
    job = args.job or tags.get("SLURM_JOB_ID", "")
    run_name = tags.get("mlflow.runName") or (re.search(r"^run_name:\s*(.+)$", meta_yaml, re.M) or [None, ""])[1]
    row = {
        "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
        "user": user, "option": option, "run_name": run_name, "job_id": job,
        "mlflow_run_id": run_dir.name,
        "arch": info["arch"], "loss": info["loss"], "key_params": info["key_params"],
        "compare_group": info["compare_group"],
        "n_params": parse_n_params(Path(args.log) if args.log else None),
        "n_out_vars": info["n_out_vars"],
        "steps": train[-1][2] + 1 if train else "",
        "s_per_step": fmt(s_per_step, 3),
        "peak_gpu_mem_mib": peak_gpu_mem(Path(args.gpu_csv) if args.gpu_csv else None),
        "train_loss_first": fmt(t_first), "train_loss_last": fmt(t_last),
        "val_loss_first": fmt(val[0][1] if val else None), "val_loss_last": fmt(val[-1][1] if val else None),
        "loss_ratio": fmt(t_last / t_first if t_first else None, 3),
        "startup_s": fmt(startup, 3),
        "train_cmd_wall_s": (str(int(args.train_end - args.train_start)) if args.train_start and args.train_end else ""),
        "curves_dir": "", "config_file": args.config or "",
    }
    curves = {"train": train, "val": val, "train_start": args.train_start}
    return row, curves


def write_curves(dest: Path, curves: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val"):
        with (dest / f"{name}_loss.csv").open("w") as fh:
            fh.write("step,value,time_s\n")
            t0 = curves["train_start"] or (curves["train"][0][0] / 1000 if curves["train"] else 0)
            for ts, v, s in curves[name]:
                fh.write(f"{s},{v:.7g},{ts / 1000 - t0:.2f}\n")


def append_row(csv_path: Path, row: dict) -> None:
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore")
    if new:
        w.writeheader()
    w.writerow(row)
    fd = os.open(csv_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o664)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError:
            pass  # filesystem without flock: a single short O_APPEND write is still safe enough
        os.write(fd, buf.getvalue().encode())
    finally:
        os.close(fd)


def cmd_add(args: argparse.Namespace) -> int:
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        if not args.job:
            print("scoreboard: give --job <slurm job id> or --run-dir", file=sys.stderr)
            return 2
        roots = args.search or [os.path.join(os.environ.get("SCRATCH", "."), "aiwcas26")]
        run_dir = find_run_dir(args.job, roots)
        if run_dir is None:
            print(f"scoreboard: no MLflow run with SLURM_JOB_ID={args.job} under {roots}", file=sys.stderr)
            return 1
    row, curves = build_row(run_dir, args)
    sb = scoreboard_dir(args.dir)
    print(f"== run       : {row['run_name']}  (option {row['option']}, {row['key_params']})")
    print(f"== train loss: {row['train_loss_first']} -> {row['train_loss_last']}  (ratio {row['loss_ratio']}, "
          f"{row['steps']} steps, {row['s_per_step']} s/step)")
    print(f"== val loss  : {row['val_loss_first']} -> {row['val_loss_last']}")
    print(f"== params {row['n_params']}, peak GPU mem {row['peak_gpu_mem_mib']} MiB, startup {row['startup_s']} s")
    if args.dry_run:
        print("== dry run: nothing written")
        return 0
    if not sb.is_dir():
        print(f"scoreboard: {sb} does not exist (instructor: `python scripts/scoreboard.py init`)", file=sys.stderr)
        return 1
    tag = f"{row['user']}_{row['job_id'] or row['mlflow_run_id'][:8]}"
    dest = sb / "curves" / tag
    write_curves(dest, curves)
    row["curves_dir"] = f"curves/{tag}"
    append_row(sb / "scoreboard.csv", row)
    print(f"== appended to {sb / 'scoreboard.csv'}")
    return 0


# ----------------------------------------------------------------------------- init / show
def cmd_init(args: argparse.Namespace) -> int:
    sb = scoreboard_dir(args.dir)
    (sb / "curves").mkdir(parents=True, exist_ok=True)
    for p in (sb, sb / "curves"):
        try:
            shutil.chown(p, group=GROUP)
        except (LookupError, PermissionError, OSError) as exc:
            print(f"warning: chgrp {GROUP} {p}: {exc}")
        os.chmod(p, 0o2775)  # setgid: new files inherit the group
    csv_path = sb / "scoreboard.csv"
    if not csv_path.exists():
        with csv_path.open("w") as fh:
            csv.DictWriter(fh, fieldnames=COLUMNS).writeheader()
    try:
        shutil.chown(csv_path, group=GROUP)
    except (LookupError, PermissionError, OSError):
        pass
    os.chmod(csv_path, 0o664)
    print(f"scoreboard ready: {csv_path}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    csv_path = scoreboard_dir(args.dir) / "scoreboard.csv"
    if not csv_path.is_file():
        print(f"no scoreboard at {csv_path}")
        return 1
    rows = list(csv.DictReader(csv_path.open()))
    cols = ["user", "option", "run_name", "n_params", "s_per_step", "peak_gpu_mem_mib",
            "train_loss_first", "train_loss_last", "val_loss_last", "loss_ratio"]
    width = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) if rows else len(c) for c in cols}
    print("  ".join(c.ljust(width[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(width[c]) for c in cols))
    print(f"({len(rows)} runs)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help=f"scoreboard directory (default $SCOREBOARD_DIR or {DEFAULT_DIR})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="append one finished run")
    a.add_argument("--job", help="Slurm job id (matched against the MLflow SLURM_JOB_ID tag)")
    a.add_argument("--run-dir", help="MLflow run directory (instead of --job)")
    a.add_argument("--search", action="append", help="where to look for runs (default $SCRATCH/aiwcas26)")
    a.add_argument("--log", help="Slurm stdout file (for the parameter count)")
    a.add_argument("--gpu-csv", help="nvidia-smi memory samples (for peak GPU memory)")
    a.add_argument("--train-start", type=float, help="unix time the training command started")
    a.add_argument("--train-end", type=float, help="unix time the training command ended")
    a.add_argument("--config", help="config file used (recorded only)")
    a.add_argument("--dry-run", action="store_true", help="print the row, write nothing")
    sub.add_parser("init", help="create the shared scoreboard directory (instructor)")
    sub.add_parser("show", help="print the scoreboard")
    args = ap.parse_args()
    return {"add": cmd_add, "init": cmd_init, "show": cmd_show}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
