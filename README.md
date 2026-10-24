# AIWCAS 2026 Anemoi tutorial

Hands-on tutorial on [Anemoi](https://anemoi.readthedocs.io), ECMWF's machine-learning weather framework, for the [AIWCAS 2026](https://dsip.fbk.eu/events/aiwcas2026/) event. It runs on the Leonardo HPC (CINECA) and uses the Anemoi packages (`anemoi-datasets`, `anemoi-graphs`, `anemoi-models`, `anemoi-training`) as pinned git submodules.

## Getting the code

```bash
git clone --recurse-submodules https://github.com/frazane/aiwcas26-anemoi-tutorial.git
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
exercises/      hour 2: "pick your setup" menu (skeleton + solution configs, compare.ipynb); start at exercises/README.md
configs/        configs used by the notebooks (e.g. 04_tiny_train.yaml)
scripts/        helper scripts (Slurm smoke test, submit.sh / train.sbatch for any config, scoreboard,
                build_shared_env.sh + anemoi_env.sh for the shared environment)
env/            requirements.in / requirements.txt (hashed lock) of the shared environment
src/
  anemoi-core/      submodule, monorepo (graphs, models, training), tag training-0.17.1
  anemoi-datasets/  submodule, tag 0.5.47
```

## Setting up the environment

### Students: use the shared environment

Everything is pre-installed in one shared, read-only environment (group `tra26_aiwcas`). In every terminal:

```bash
source /leonardo_scratch/fast/tra26_aiwcas/aiwcas26/envs/anemoi/bin/activate
```

No `module load` and no installing. It contains Python 3.11.7 (the `python/3.11.7` module's interpreter), torch 2.10.0+cu126, torchvision, the released `anemoi-datasets 0.5.47`, `anemoi-graphs 0.9.8` (with `trimesh`), `anemoi-models 0.19.0`, `anemoi-training 0.17.1` (the versions the `src/` submodules are pinned to), `datashader`, and JupyterLab, matplotlib, cartopy and plotly for the notebooks. The exact versions are in [`env/requirements.txt`](env/requirements.txt).

The Slurm scripts (`scripts/submit.sh`, `scripts/train.sbatch`, `scripts/smoke_test_gpu.sbatch`, `scripts/train_tiny.sbatch`) use the same environment by themselves, so there is nothing to activate for them (see [How jobs use the environment](#how-jobs-use-the-environment)).

### Notebooks: VS Code or JupyterLab

CINECA has no Jupyter service for Leonardo (its [interactive computing service](https://docs.hpc.cineca.it/services/interactive_computing.html) is Galileo100 only), so notebooks run on a **login node**. Login nodes kill any process after 10 min of CPU time, kernels included: restart the kernel if it dies, and leave training to the Slurm jobs.

Register the shared environment as a Jupyter kernel once (it is written to `~/.local/share/jupyter/kernels/`):

```bash
/leonardo_scratch/fast/tra26_aiwcas/aiwcas26/envs/anemoi/bin/python -m ipykernel install --user \
    --name aiwcas26-anemoi --display-name "Python (aiwcas26 anemoi)"
```

**VS Code (recommended).** Connect with the *Remote - SSH* extension to `login.leonardo.cineca.it` (with your usual 2FA SSH certificate) and open your clone of this repository. Install the *Python* and *Jupyter* extensions on the remote side when VS Code offers it. Then:

- for notebooks, open an `.ipynb`, click *Select Kernel* (top right), then *Jupyter Kernel...* → **Python (aiwcas26 anemoi)**. Alternatively, choose *Python Environments...* → *Enter interpreter path*, as below;
- for scripts and the terminal, run *Python: Select Interpreter* → *Enter interpreter path...* and give
  `/leonardo_scratch/fast/tra26_aiwcas/aiwcas26/envs/anemoi/bin/python`.

**JupyterLab in the browser.** On the login node, pick a free port, note which login node you are on, and start JupyterLab:

```bash
source /leonardo_scratch/fast/tra26_aiwcas/aiwcas26/envs/anemoi/bin/activate
hostname                                    # e.g. login07.leonardo.local
jupyter lab --no-browser --port=8$((RANDOM % 900 + 100))    # prints a URL with a token
```

Then, on your laptop, tunnel to **that** login node: `loginNN.leonardo.local` is reachable as `loginNN-ext.leonardo.cineca.it`, while `login.leonardo.cineca.it` may send you to another one. Open the printed `http://localhost:<port>/lab?token=...` URL:

```bash
ssh -N -L <port>:localhost:<port> <username>@login07-ext.leonardo.cineca.it
```

Pick the kernel **Python (aiwcas26 anemoi)** (or the default *Python 3*, which is the same environment when JupyterLab was started from it).

### How jobs use the environment

The environment exists twice, with identical contents:

- `/leonardo_scratch/fast/tra26_aiwcas/aiwcas26/envs/anemoi/`: a normal venv on Lustre (`$FAST`, SSD pool), for the login node, Jupyter kernels, VS Code and `anemoi-training config validate`;
- `/leonardo_scratch/fast/tra26_aiwcas/aiwcas26/envs/anemoi.sqfs`: a squashfs image of the same directory, for jobs.

Importing torch and anemoi means opening tens of thousands of small files, which on Lustre took 3–4 min of every 6-min training job. `scripts/anemoi_env.sh` (sourced by the Slurm scripts) instead mounts the image with `squashfuse_ll` (shipped with Singularity, in `/usr/libexec/singularity/bin/`) at `/tmp/aiwcas26-$USER-$SLURM_JOB_ID/` on the compute node and activates it from there. The venv is relocatable, so it works at that path. The mount goes away with the job. In a test run, startup up to the first training step dropped from 286 s to 21 s.

Knobs (environment variables, passed through by `scripts/submit.sh`):

| variable | default | meaning |
|---|---|---|
| `ANEMOI_ENV_MODE` | `auto` | `auto`: mount the image if there is one next to the venv, falling back to the plain venv with a warning if the mount fails; `sqfs`: the image or fail; `venv`: always the plain venv |
| `ANEMOI_VENV` | the shared env | another venv, e.g. `ANEMOI_VENV=$SCRATCH/envs/anemoi scripts/submit.sh ...` (without a `<venv>.sqfs` image next to it, it is used as a plain venv) |
| `ANEMOI_SQFS` | `<venv>.sqfs` | another image |

### Instructors: building or updating the shared environment

```bash
scripts/build_shared_env.sh --name anemoi-v2 --activate     # on a login node, ~10 min
```

This creates the venv `$FAST/aiwcas26/envs/<name>` with `uv venv --relocatable` from the hashed lock `env/requirements.txt` (non-editable; torch from the PyTorch cu126 index). It then pre-compiles all bytecode, registers a Jupyter kernel inside the env, makes everything group-readable and not group-writable (setgid dirs), and writes `<name>.sqfs` (gzip, striped over 8 SSD OSTs). `--activate` then points the stable names `anemoi` and `anemoi.sqfs` at the new build. Builds are immutable: use a new `--name` for every rebuild, and running jobs keep the image they mounted. To change versions, edit `env/requirements.in` and add `--lock`.

### Fallback: your own environment

If the shared environment is unavailable, build your own in `$SCRATCH` (editable installs from the submodules; this is how it was done before the shared env). Anemoi needs Python >=3.11,<3.14 and torch >=2.3,<2.11. The system `python3` is 3.6, and the `cineca-ai` module ships torch 2.2, which is too old, so do not use it. There is no conda.

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

Then use it in jobs with `ANEMOI_VENV=$SCRATCH/envs/anemoi` (it runs as a plain venv: in the test run, startup up to the first training step took 315 s instead of 21 s).

Notes:

- Install torch from the PyTorch index first, otherwise PyPI torch gets pulled in.
- `UV_LINK_MODE=copy` is needed because the uv cache and the venv are on different filesystems.
- Keep caches out of `$HOME` (50 GB quota).
- Result: torch 2.10.0+cu126, torch-geometric 2.8.1, pytorch-lightning 2.6.6.
- Versions like `anemoi-models 0.19.0.post3` are expected for editable installs (commits since the tag in the monorepo).
- Importing anemoi from a Lustre venv is slow (30 s to several minutes on a login node, depending on load).

## GPU smoke test

Checks that the environment sees a GPU and that torch, torch-geometric and all Anemoi packages import and run.

```bash
sbatch scripts/smoke_test_gpu.sbatch
# the shared env as a plain venv, or another env:
ANEMOI_ENV_MODE=venv sbatch scripts/smoke_test_gpu.sbatch
ANEMOI_VENV=/path/to/venv sbatch scripts/smoke_test_gpu.sbatch
```

It uses account `tra26_aiwcas`, partition `boost_usr_prod`, the debug QOS `boost_qos_dbg`, 1 GPU and 8 cores. Output goes to `anemoi-smoke-<jobid>.out` in the directory you submitted from; a successful run ends with `SMOKE TEST PASSED`. With the shared env it takes about 15 s of wall time (job 62147541: 11 s, imports 6.4 s from the squashfs), plus queueing: Booster is often nearly full. The GPU driver is 535 (CUDA 12.2 natively); the cu126 wheels work through CUDA minor-version compatibility.

## Session outline

TODO
