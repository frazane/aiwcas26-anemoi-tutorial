# AIWCAS 2026 Anemoi tutorial

Hands-on tutorial on [Anemoi](https://anemoi.readthedocs.io), ECMWF's machine-learning weather framework, for the [AIWCAS 2026](https://dsip.fbk.eu/events/aiwcas2026/) event. It runs on the Leonardo HPC (CINECA) and uses the Anemoi packages (`anemoi-datasets`, `anemoi-graphs`, `anemoi-models`, `anemoi-training`).

## Getting started

In a terminal on a Leonardo login node (`ssh <username>@login.leonardo.cineca.it`, or a VS Code Remote-SSH terminal):

```bash
git clone --recurse-submodules https://github.com/frazane/aiwcas26-anemoi-tutorial.git
cd aiwcas26-anemoi-tutorial
source setup.sh
```

`setup.sh` checks your account, activates the tutorial's Python environment (everything is pre-installed:
nothing to install) and ends with **"You're ready"**. If it prints a line starting with `!!`, do what that line
says and run `source setup.sh` again. In every **new terminal**, `cd` into the repository and run
`source setup.sh` again (it is quick and safe to repeat).

**Hour 1: the notebooks.** Open the repository folder in VS Code (*Remote - SSH* to `login.leonardo.cineca.it`,
then *File → Open Folder...*), install the *Python* and *Jupyter* extensions when VS Code offers them, and open
`notebooks/01_datasets.ipynb`. The kernel **Python (aiwcas26 anemoi)** is selected automatically (if VS Code asks,
pick it under *Select Kernel → Jupyter Kernel...*).

**Hour 2: train your own model.** Start at [`exercises/README.md`](exercises/README.md). Everything you run there
goes through one command, `aiwcas`:

| command | what it does |
|---|---|
| `aiwcas validate <config.yaml>` | checks a config in 5–20 s: unfilled `???`, the anemoi schema, and variable names |
| `aiwcas train <config.yaml> [--name N]` | validates, then trains it on one GPU (a Slurm job of a few minutes) |
| `aiwcas status` | your jobs: waiting for a GPU, running, finished |
| `aiwcas log [jobid]` | follows the log of your latest job (Ctrl-C stops watching, not the job) |
| `aiwcas plot [jobid]` | the loss curve of your latest job |
| `aiwcas scoreboard` | everybody's results; `exercises/compare.ipynb` plots them |
| `aiwcas help` | all commands, with examples |

Two things to know about Leonardo: notebooks run on the shared **login node**, which stops any process (a
notebook kernel too) after 10 minutes of CPU time, so restart the kernel if it dies; and **training runs on the GPU
nodes**, through `aiwcas train`, never on the login node.

## Layout

```
README.md       this file
setup.sh        `source setup.sh`: sets up a terminal (environment, `aiwcas` command, Jupyter kernel, checks)
bin/aiwcas      the student command line (validate / train / status / log / plot / scoreboard / check)
AGENTS.md       rules and gotchas for coding agents
notebooks/      tutorial notebooks (hour 1)
exercises/      hour 2: "pick your setup" menu (skeleton + solution configs, compare.ipynb); start at exercises/README.md
configs/        configs used by the notebooks (e.g. 04_tiny_train.yaml)
scripts/        what `aiwcas` runs (submit.sh / train.sbatch, validate_config.py, plot_loss.py, scoreboard.py,
                smoke_test_gpu.sbatch) and instructor tools (build_shared_env.sh + anemoi_env.sh, ...)
env/            requirements.in / requirements.txt (hashed lock) of the shared environment
.vscode/        workspace settings: the shared interpreter, `aiwcas` on the terminal PATH
src/
  anemoi-core/      submodule, monorepo (graphs, models, training), tag training-0.17.1
  anemoi-datasets/  submodule, tag 0.5.47
```

## For instructors and advanced users

Students do not need anything below.

### What `setup.sh` sets up

`source setup.sh [-q]` (bash only; it refuses to run when executed instead of sourced and never exits your shell):

- **checks**, each with a fix when it fails: on a login node, `$SCRATCH` set, member of group `tra26_aiwcas`, the
  shared env readable and its Python 3.11.7 (the `python/3.11.7` module's interpreter) starting, the tutorial data
  readable; it also fetches the git submodules if the clone has none;
- **activates** the shared env (the plain venv: on the login node there is no image to mount) and puts `bin/` on `PATH`;
- **registers** the Jupyter kernel `aiwcas26-anemoi` ("Python (aiwcas26 anemoi)") in `~/.local/share/jupyter/kernels/`,
  rewriting it only when it changed; the kernel gets `AIWCAS_DATA`, `AIWCAS_WORKDIR` and the MLflow variables below in its environment;
- **creates** `$SCRATCH/aiwcas26/{logs,training,graphs,hour2/graphs}`;
- **exports** the paths of the tutorial. They are defined in `setup.sh` only (change them there), and an existing
  value in the environment wins:

| variable | default | used by |
|---|---|---|
| `AIWCAS_REPO` | the clone `setup.sh` is in | `aiwcas` |
| `AIWCAS_ENV` | `/leonardo_scratch/fast/tra26_aiwcas/aiwcas26/envs/anemoi` (or `$ANEMOI_VENV`) | activation, kernel, `ANEMOI_VENV` for jobs |
| `AIWCAS_DATA` | `/leonardo_work/tra26_aiwcas/a08trd25/data` (`anemoi/` zarr stores, `cartopy/` shapefiles) | data check; configs can read it as `${oc.env:AIWCAS_DATA}`, notebooks as `os.environ["AIWCAS_DATA"]` |
| `AIWCAS_WORKDIR`, `AIWCAS_LOGS` | `$SCRATCH/aiwcas26`, `$SCRATCH/aiwcas26/logs` | Slurm logs, loss plots |
| `MLFLOW_ALLOW_FILE_STORE`, `MLFLOW_DISABLE_AGENT_HINT` | `true`, `1` | offline MLflow in notebooks |

Jobs inherit all of them (`scripts/submit.sh` uses `sbatch --export=ALL`). `bin/aiwcas` sources `setup.sh -q` itself
when the terminal was not set up, so it also works from the VS Code terminal (whose `PATH` includes `bin/`, see
`.vscode/settings.json`). The configs and notebooks still spell the data path out; moving the data means changing
`AIWCAS_DATA` and replacing `/leonardo_work/tra26_aiwcas/a08trd25/data` by `${oc.env:AIWCAS_DATA}` in the configs and
by `os.environ["AIWCAS_DATA"]` in the notebooks.

To set up every terminal automatically, a student can add `source <clone>/setup.sh -q` to `~/.bashrc` (`setup.sh`
prints the exact line; it never edits `~/.bashrc` itself).

### The shared environment

Everything is pre-installed in one shared, read-only environment (group `tra26_aiwcas`). It contains Python 3.11.7 (the `python/3.11.7` module's interpreter), torch 2.10.0+cu126, torchvision, the released `anemoi-datasets 0.5.47`, `anemoi-graphs 0.9.8` (with `trimesh`), `anemoi-models 0.19.0`, `anemoi-training 0.17.1` (the versions the `src/` submodules are pinned to), `datashader`, and JupyterLab, matplotlib, cartopy and plotly for the notebooks. The exact versions are in [`env/requirements.txt`](env/requirements.txt). Without `setup.sh`: `source /leonardo_scratch/fast/tra26_aiwcas/aiwcas26/envs/anemoi/bin/activate`.

The Slurm scripts (`scripts/submit.sh`, `scripts/train.sbatch`, `scripts/smoke_test_gpu.sbatch`, `scripts/train_tiny.sbatch`) use the same environment by themselves, so there is nothing to activate for them (see [How jobs use the environment](#how-jobs-use-the-environment)).

### JupyterLab in the browser

CINECA has no Jupyter service for Leonardo (its [interactive computing service](https://docs.hpc.cineca.it/services/interactive_computing.html) is Galileo100 only), so notebooks run on a **login node**, in VS Code (above) or JupyterLab.

If you prefer JupyterLab to VS Code: in a terminal where you ran `source setup.sh`, pick a free port, note which login node you are on, and start JupyterLab:

```bash
hostname                                    # e.g. login07.leonardo.local
jupyter lab --no-browser --port=8$((RANDOM % 900 + 100))    # prints a URL with a token
```

Then, on your laptop, tunnel to **that** login node: `loginNN.leonardo.local` is reachable as `loginNN-ext.leonardo.cineca.it`, while `login.leonardo.cineca.it` may send you to another one. Open the printed `http://localhost:<port>/lab?token=...` URL:

```bash
ssh -N -L <port>:localhost:<port> <username>@login07-ext.leonardo.cineca.it
```

The notebooks select the kernel **Python (aiwcas26 anemoi)** by themselves (the default *Python 3* is the same environment when JupyterLab was started after `source setup.sh`).

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

### Building or updating the shared environment

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

### GPU smoke test

Checks that the environment sees a GPU and that torch, torch-geometric and all Anemoi packages import and run.
`aiwcas check` submits it (log in `$SCRATCH/aiwcas26/logs/`) and follows the log. By hand:

```bash
sbatch scripts/smoke_test_gpu.sbatch
# the shared env as a plain venv, or another env:
ANEMOI_ENV_MODE=venv sbatch scripts/smoke_test_gpu.sbatch
ANEMOI_VENV=/path/to/venv sbatch scripts/smoke_test_gpu.sbatch
```

It uses account `tra26_aiwcas`, partition `boost_usr_prod`, the debug QOS `boost_qos_dbg`, 1 GPU and 8 cores. With plain `sbatch`, output goes to `anemoi-smoke-<jobid>.out` in the directory you submitted from; a successful run ends with `SMOKE TEST PASSED`. With the shared env it takes about 15 s of wall time (job 62147541: 11 s, imports 6.4 s from the squashfs), plus queueing: Booster is often nearly full. The GPU driver is 535 (CUDA 12.2 natively); the cu126 wheels work through CUDA minor-version compatibility.

## Session outline

TODO
