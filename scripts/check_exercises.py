#!/usr/bin/env python
"""Instructor tool: validate (and optionally CPU-smoke-train) many configs in ONE Python process.

`anemoi-training config validate` pays 1-3 min of imports per call on a login node. This script
imports anemoi once and then loops over the configs, so checking every skeleton and solution
takes about as long as one `validate`.

    python scripts/check_exercises.py                      # validate every exercises/*/*.yaml
    python scripts/check_exercises.py exercises/C_ensemble_crps/solution.yaml --cpu-steps 3
    python scripts/check_exercises.py a.yaml b.yaml -o training.max_steps=10

Skeletons are expected to FAIL (they contain ??? placeholders); the script prints the first
line of each error so you can check it is understandable.

--cpu-steps N trains N steps on the CPU (32-bit, single-device strategy, 1 validation batch,
outputs under $SCRATCH/aiwcas26/hour2/cpu_check). Be considerate: the login node is shared.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("HYDRA_FULL_ERROR", "1")

REPO = Path(__file__).resolve().parent.parent

CPU_OVERRIDES = [
    "system.hardware.accelerator=cpu",
    "training.precision=32-true",
    "training.strategy._target_=anemoi.training.distributed.single.SingleDeviceStrategy",
    "dataloader.num_workers.training=2",
    "dataloader.num_workers.validation=1",
    "dataloader.limit_batches.validation=1",
    "training.optimization.lr_scheduler.warmup_t=1",
    "diagnostics.log.interval=1",
    "diagnostics.progress_bar.refresh_rate=1",
    "diagnostics.checkpoint.every_n_epochs.save_frequency=null",
    "diagnostics.checkpoint.every_n_epochs.num_models_saved=0",
]


def compose_cfg(path: Path, overrides: list[str]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(path.parent)):
        return compose(config_name=path.stem, overrides=overrides)


def first_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: " + (" | ".join(t.strip() for t in text[:6]) if text else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*", help="yaml files (default: exercises/*/*.yaml and variants)")
    ap.add_argument("-o", "--override", action="append", default=[], help="extra Hydra override (repeatable)")
    ap.add_argument("--cpu-steps", type=int, default=0, help="also train N steps on the CPU")
    args = ap.parse_args()

    paths = [Path(p).resolve() for p in args.configs] or sorted(
        Path(p) for p in glob.glob(str(REPO / "exercises" / "*" / "**" / "*.yaml"), recursive=True)
    )

    t0 = time.time()
    from omegaconf import OmegaConf

    from anemoi.training.schemas.base_schema import BaseSchema

    print(f"[imports {time.time() - t0:.0f} s]", flush=True)

    results = []
    for path in paths:
        t = time.time()
        overrides = list(args.override)
        try:
            cfg = compose_cfg(path, overrides)
            OmegaConf.resolve(cfg)
            BaseSchema(**cfg)
            status, msg = "VALID", ""
        except Exception as exc:  # noqa: BLE001
            status, msg = "INVALID", first_line(exc)
        print(f"{status:8s} {path.relative_to(REPO) if path.is_relative_to(REPO) else path}  ({time.time() - t:.1f} s)")
        if msg:
            print(f"         -> {msg[:600]}")
        if status == "VALID" and args.cpu_steps:
            from anemoi.training.train.train import AnemoiTrainer

            out = Path(os.environ["SCRATCH"]) / "aiwcas26" / "hour2" / "cpu_check" / path.parent.name
            ov = CPU_OVERRIDES + [f"training.max_steps={args.cpu_steps}", f"system.output.root={out}"] + overrides
            t = time.time()
            try:
                trainer = AnemoiTrainer(compose_cfg(path, ov))
                n_params = sum(p.numel() for p in trainer.model.parameters())
                trainer.train()
                status = "TRAINED"
                msg = f"{n_params / 1e6:.2f} M params, {args.cpu_steps} CPU steps"
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                status, msg = "TRAINFAIL", first_line(exc)
            print(f"{status:8s} {path.name}: {msg} ({time.time() - t:.0f} s)", flush=True)
        results.append((path, status))

    print(f"[total {time.time() - t0:.0f} s]")
    return 0 if all(s != "TRAINFAIL" for _, s in results) else 1


if __name__ == "__main__":
    sys.exit(main())
