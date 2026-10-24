# AGENTS.md

Project description, quickstart and layout: see [README.md](README.md). Not repeated here.

## Agent rules

- **Minimise project allocation.** Account `tra26_aiwcas` has 5000 core-hours in total, shared by ~35 users (valid until 2026-11-04). Billing is per CPU core.
  - Prefer the login node (`lrd_all_serial` runs CPU work there) over compute jobs.
  - If a GPU is needed: debug QOS `boost_qos_dbg` (30 min max, 2 running / 4 submitted jobs per user), 1 GPU, short walltime, few cores (e.g. 8 per GPU).
- **Never edit submodule contents** (`src/anemoi-core`, `src/anemoi-datasets`). Bump pins with `git -C src/<pkg> checkout <tag>`, then stage the pointer in the parent repo.
- **anemoi-core is a monorepo with per-package tags** (`training-X`, `models-X`, `graphs-X`). Pin to the latest release tag. Current pin: `training-0.17.1` (== `graphs-0.9.8`, includes `models-0.19.0`). anemoi-datasets is pinned at `0.5.47`.
- Compute nodes have no internet: install and download everything on the login node.
- `$FAST/data/arco-it-dpc-sri/` (organizer radar data) is read-only. Do not modify it.

## Gotchas

Append new entries as `- **Short title.** What happens, and what to do.`

- **Python version.** Anemoi needs Python >=3.11,<3.14 and torch >=2.3,<2.11. System `python3` is 3.6; use `module load python/3.11.7`.
- **Do not use `cineca-ai`** (`profile/deeplrn`): its torch 2.2 is too old. No conda; Singularity exists but is not used.
- **Install order.** Install torch from the PyTorch index (`--index-url https://download.pytorch.org/whl/cu126`) before the Anemoi packages, otherwise PyPI torch gets pulled in.
- **uv cache.** `UV_LINK_MODE=copy` is required (cache and venv are on different filesystems). Keep caches out of `$HOME` (50 GB quota, backup currently suspended): `UV_CACHE_DIR=$SCRATCH/.cache/uv`.
- **No internet on compute nodes.** Login nodes have it. Pre-download everything there.
- **CUDA driver.** Driver 535.274.02 is natively CUDA 12.2; cu126 wheels work via minor-version compatibility. Verified on A100-SXM-64GB.
- **Slow imports.** Importing anemoi from a Lustre venv takes ~28 s on the login node.
- **Odd versions.** `anemoi-models 0.19.0.post3` and similar are expected: setuptools_scm counts commits since the models tag in the monorepo.
- **Storage.** `$HOME` 50 GB (no backup currently); `$WORK=/leonardo_work/tra26_aiwcas` 1 TB group-shared; `$FAST=/leonardo_scratch/fast/tra26_aiwcas` group-shared; `$SCRATCH` is per-user and files older than 40 days are purged.
- **Environment location.** The final student env location is TBD (likely shared in `$WORK`). Default scripts use `$SCRATCH/envs/anemoi`; override with `ANEMOI_VENV`.
- **Queue.** Booster (`boost_usr_prod`, 4xA100 64GB, 32 cores/node) is often nearly full; there is no event reservation yet.
- **Slurm output.** `scripts/smoke_test_gpu.sbatch` writes `%x-%j.out` in the submit directory, so submit from somewhere you do not mind littering.
- **`TriNodes` needs `trimesh`.** The plain `anemoi-graphs` install does not pull it in, but the default training graph (`graph: multi_scale`) uses `TriNodes`. Install it with `uv pip install "trimesh>=4.1"` (the `[tri]` extra). `healpy` (`[healpix]`) is already in the venv; `h3` (`[hex]`, for `HexNodes`) is not.
- **Graph build time is mostly imports.** On a login node, `anemoi-graphs create` for O96 + `TriNodes` r=5 takes 1.5–3.5 min wall, almost all of it Python imports from Lustre. In an already-warm kernel the same `GraphCreator(...).create()` takes about 15–25 s the first time (lazy imports) and about 5 s after that. O48 + r=4 builds in about 1 s.
- **`torch-cluster` warning.** anemoi-graphs logs that installing `torch-cluster` would speed up edge building. This is harmless; the sklearn fallback is fast enough for O96.
- **`anemoi-graphs inspect` 3D HTML needs internet in the viewer.** `graph.html` (three.js) loads its library and the Earth texture from CDNs, so open it in a laptop browser, not on the cluster. It embeds every node and edge, so a full O96/r5 graph gives about 12 MB; use a smaller graph or `--area` for demos.
- **`MultiScaleEdges(scale_resolutions=<int>)` means levels 1..r**, not 0..r as its docstring says (`list(range(1, r + 1))`).
- **Building a model outside anemoi-training.** Compose only the groups you need (`compose(overrides=["+model=graphtransformer", "+data=zarr"])` with `initialize_config_dir` on `src/anemoi-core/training/src/anemoi/training/config`), then call `OmegaConf.resolve(cfg)` before `AnemoiModelInterface(config=cfg, ...)`. Unresolved, the model's internal Hydra `instantiate` fails with `Interpolation key 'model.num_channels' not found`. The interface also needs `metadata={"metadata_inference": {"data": {}}}` and `data_indices`/`statistics` as dicts keyed by dataset name (`"data"`). `model.model(...)` and `predict_step(...)` take a dict `{"data": tensor}`, not a bare tensor.
- **CPU overrides for anemoi-models.** The default GraphTransformer backend `triton` silently falls back to `pyg` on CPU (with a warning); set `graph_attention_backend=pyg` to make it explicit. `TransformerProcessor` needs `attention_implementation=scaled_dot_product_attention` (flash-attn is GPU only).
- **`model/transformer.yaml` is incomplete in models-0.19.0.** Standalone it lacks `latent_aggregator` and the decoder `num_channels` (add `+model.latent_aggregator._target_=anemoi.models.layers.aggregator.SumAggregator` and `+model.decoders.0.mapper.num_channels=${model.num_channels}`). Its graph must have no `hidden→hidden` edges (`AssertionError: Edge attributes must be provided`); use `graph=encoder_decoder_only` or `del graph["hidden","to","hidden"]`.
- **Data vs model indices.** `IndexCollection.data.*.name_to_index` gives positions in the full dataset (e.g. 101 vars); `.model.input/.model.output.name_to_index` gives positions in the tensors the model sees (99 in / 88 out with `tp`, `cp` diagnostic). Mixing them silently picks the wrong variable.
- **NaNs and pre-processors.** `Processors` checks for NaNs only on its first call; after that NaNs pass silently and spread through the graph (one land-masked field made 75% of all outputs NaN). List the imputer *before* the normaliser in `data.datasets.data.processors` (config order = application order).
- **Small CPU models are dominated by trainable graph features.** With `num_channels=32` about 97% of the ~2.6 M parameters are the 8 trainable features per node/edge (`node/edge_trainable_parameters`) on the O96/r5 graph; the stock `graphtransformer.yaml` has ~249 M, 88% in the processor. Building it on CPU takes ~5 s and ~1 GB RAM.
- **`anemoi-training train` needs `datashader`.** The training package imports it unconditionally (via the plotting callbacks), even with `diagnostics.plot.callbacks: []`. Installed `datashader==0.19.1` (+ colorcet, multipledispatch, param, pyct) in the venv. Set `diagnostics.plot.settings.datashader: False` anyway if you keep plot callbacks.
- **MLflow offline needs `MLFLOW_ALLOW_FILE_STORE=true`.** mlflow-skinny 3.17 refuses the `./mlruns` file store ("in maintenance mode") that `diagnostics.log.mlflow.offline: True` uses. Export the variable (done in `scripts/train_tiny.sbatch` and notebook 04). Metrics then land as plain text files in `<output.root>/logs/mlflow/<exp>/<run>/metrics/` ("timestamp value step").
- **`train --config-path` must be absolute.** Hydra resolves a relative `--config-path` against the `anemoi.training.train` package ("Primary config module ... not found"). Use `--config-path $PWD/configs`. `anemoi-training config validate` accepts relative paths.
- **Hydra writes `outputs/<date>/<time>/` in the cwd** for every `anemoi-training train`. Set `hydra.run.dir` (`configs/04_tiny_train.yaml` uses `$SCRATCH/aiwcas26/training/hydra/...`) or you litter the repo.
- **`training.precision=32` fails validation.** The schema wants a string; use `32-true`, `16-mixed` or `bf16-mixed`. Good demo of a pydantic error.
- **No DDP in Jupyter.** Lightning refuses `DDPGroupStrategy` (the default, even for 1 device) in an interactive session. Override `training.strategy._target_=anemoi.training.distributed.single.SingleDeviceStrategy` (the extra DDP keys are dropped by the class). Scripts and `srun` keep the default.
- **Short GPU jobs are mostly startup.** For the 1-GPU tiny run (O96, 2.3 M params) the job took 5.5 min: ~3–4 min Python imports + config/graph/model setup on the compute node, ~1 min for 500 steps (~9 it/s). With 256 channels, 8 layers, rollout 2: 6.5 min job, ~6.5 it/s, 4 GB GPU memory. Login-node import of the training stack alone took 1–3 min under load.
- **Tiny sample dataset splits.** The O96 sample has 39 dates (2017-01-01T06..2017-01-10T18). `start: 2017-01-01, end: 2017-01-08` gives 31 dates = 29 training samples with multistep 2 + rollout 1; `2017-01-09..10` gives 8 dates = 6 validation samples. Every extra rollout step removes one sample.
