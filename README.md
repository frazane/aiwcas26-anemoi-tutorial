# AIWCAS 2026 Anemoi tutorial

Hands-on tutorial on [Anemoi](https://anemoi.readthedocs.io), ECMWF's machine-learning weather framework, for the [AIWCAS 2026](https://dsip.fbk.eu/events/aiwcas2026/) event. It runs on the Leonardo HPC (CINECA) and uses the Anemoi packages (`anemoi-datasets`, `anemoi-graphs`, `anemoi-models`, `anemoi-training`) as pinned git submodules.

## Getting the code

```bash
git clone --recurse-submodules <REPO_URL>
cd aiwcas26-anemoi-tutorial
```

If you forgot the flag:

```bash
git submodule update --init
```

## Layout

```
README.md       this file
AGENTS.md       rules and gotchas for coding agents
notebooks/      tutorial notebooks
scripts/        helper scripts (e.g. Slurm smoke test)
src/
  anemoi-core/      submodule, monorepo (graphs, models, training), tag training-0.17.1
  anemoi-datasets/  submodule, tag 0.5.47
```

## Setting up the environment

Anemoi needs Python >=3.11,<3.14 and torch >=2.3,<2.11. The system `python3` is 3.6, and the `cineca-ai` module ships torch 2.2, which is too old, so do not use it. There is no conda.

Do all of this on a **login node**: compute nodes have no internet.

```bash
module load python/3.11.7
export UV_CACHE_DIR=$SCRATCH/.cache/uv UV_LINK_MODE=copy
python3 -m venv $SCRATCH/envs/anemoi
source $SCRATCH/envs/anemoi/bin/activate
pip install uv
uv pip install "torch>=2.3,<2.11" torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -e src/anemoi-datasets -e "src/anemoi-core/graphs[tri]" -e src/anemoi-core/models -e src/anemoi-core/training
uv pip install datashader   # imported unconditionally by `anemoi-training train`
uv pip install jupyterlab ipykernel nbconvert matplotlib cartopy plotly   # notebooks
```

Notes:

- Install torch from the PyTorch index first, otherwise PyPI torch gets pulled in.
- `UV_LINK_MODE=copy` is needed because the uv cache and the venv are on different filesystems.
- Keep caches out of `$HOME` (50 GB quota).
- Result: torch 2.10.0+cu126, torch-geometric 2.8.1, pytorch-lightning 2.6.6.
- Versions like `anemoi-models 0.19.0.post3` are expected for editable installs (commits since the tag in the monorepo).
- Importing anemoi from a Lustre venv is slow (~30 s on a login node).

The final environment location for students is still TBD (likely a shared env in `$WORK`); until then build your own in `$SCRATCH`, or point to the shared one with `ANEMOI_VENV`.

## GPU smoke test

Checks that the environment sees a GPU and that torch, torch-geometric and all Anemoi packages import and run.

```bash
sbatch scripts/smoke_test_gpu.sbatch
# or, with another env:
ANEMOI_VENV=/path/to/venv sbatch scripts/smoke_test_gpu.sbatch
```

It uses account `tra26_aiwcas`, partition `boost_usr_prod`, the debug QOS `boost_qos_dbg`, 1 GPU and 8 cores. Output goes to `anemoi-smoke-<jobid>.out` in the directory you submitted from; a successful run ends with `SMOKE TEST PASSED`. Expect about 2 minutes of wall time (plus queueing: Booster is often nearly full). The GPU driver is 535 (CUDA 12.2 natively); the cu126 wheels work through CUDA minor-version compatibility.

## Session outline

TODO
