# Hour 2 · Pick your setup

You have trained the hour-1 model. Now pick **one** option, turn its skeleton config into a working one,
train it on one GPU for a few minutes, and see how it compares with the rest of the room.

| | option | what you train | anemoi concept | difficulty | GPU job (tested) |
|---|---|---|---|---|---|
| **0** | [make the base yours](0_make_it_yours/) | the hour-1 GNN with your variables, loss weights, input history and rollout | data selection, forcing / diagnostic / prognostic variables, loss scalers, rollout | ★ | ~6 min |
| **A** | [processor zoo](A_processor_zoo/) | the same forecaster with a GNN, GraphTransformer, Transformer or point-wise MLP processor | the architecture is a config choice: encoder → **processor** → decoder | ★ | ~6 min |
| **C** | [ensemble CRPS](C_ensemble_crps/) | a 4-member ensemble trained with the almost-fair CRPS (AIFS-ENS recipe) | task ≠ training method: `EnsembleTraining`, noise-conditioned model, CRPS | ★★ | ~6 min |
| **B1** | [Italy LAM](B1_lam/) | a regional model over Italy (CERRA, 22 km) with a ring of global ERA5 as boundary | dataset `cutout`, limited-area graph, output mask / LAM loss | ★★★ | ~6 min |

Every option uses the 10-day O96 ERA5 sample from hour 1 (29 training and 6 validation samples); B1 adds the
same 10 days of CERRA over Italy. We watch the loss go down, we do not build a useful forecast model. **Not sure? Take A** and agree with your neighbours
on who runs which processor, so the room covers all four. **0** is also the fallback if anything goes wrong.

Reference GPU dry runs (1 × A100, `boost_qos_dbg`, 8 cores; the solution configs as shipped):

| option | job | wall time | of which startup | steps | it/s (steady) | peak GPU mem | train loss first → last | val loss first → last |
|---|---|---|---|---|---|---|---|---|
| 0 solution (GNN, 20 outputs, rollout 2) | 62116752 | 6 min 07 s | 4 min 46 s | 500 | 8.7 | 2.3 GiB | 0.105 → 0.059 | 0.101 → 0.071 |
| A solution (GraphTransformer, triton) | 62116755 | 6 min 16 s | 4 min 53 s | 500 | 9.5 | 2.3 GiB | 0.145 → 0.035 | 0.073 → 0.039 |
| C solution (GT-ens, 4 members, afCRPS) | 62117462 | 6 min 04 s | 4 min 41 s | 400 | 9.2 | 4.7 GiB | 0.295 → 0.097 | 0.204 → 0.106 |
| B1 solution (LAM, GT, thinning 4, hidden r7) | 62127960 | 5 min 59 s | 5 min 07 s | 500 | 15.9 | 1.9 GiB | 0.510 → 0.122 | 0.141 → 0.124 |

"Startup" is everything before the first logged step: Python imports from the shared filesystem
(about 3–4 min), config, data, graph and model setup. **Training itself takes about a minute.**
At this size the 4 ensemble members of C cost almost nothing extra per training step (9.2 it/s, like A);
they double the GPU memory, and C's validation passes are slower. B1 is the smallest model (0.57 M parameters,
3948 points, 5 outputs), so its steps are the fastest; its loss is an MSE over Italy only, not comparable with
the others.

Not GPU-tested yet: A's `variants/transformer.yaml` and `variants/point_wise.yaml` (both train on CPU; budget
kept the GPU dry runs to one per option). In the runs above the graph file already existed; a student's first
run also builds it (a few seconds for these small graphs).

## Steps (the same for every option)

All commands run from the repository root, **on the login node**.

```bash
# 0. once per terminal
cd ~/aiwcas26-anemoi-tutorial            # wherever you cloned it
module load python/3.11.7
source ${ANEMOI_VENV:-$SCRATCH/envs/anemoi}/bin/activate

# 1. copy the skeleton of your option (here A); mine*.yaml is git-ignored
cp exercises/A_processor_zoo/skeleton.yaml exercises/A_processor_zoo/mine.yaml

# 2. fill the TODOs: every ??? in mine.yaml (the comments next to each one tell you what to put)

# 3. validate (1–3 min, mostly imports). Repeat until it ends with "Config files validated."
anemoi-training config validate --config-path exercises/A_processor_zoo --config-name mine

# 4. submit one 1-GPU job (10 min limit, debug queue)
scripts/submit.sh exercises/A_processor_zoo/mine.yaml
#    optional: a run name and Hydra overrides on the command line
scripts/submit.sh exercises/A_processor_zoo/mine.yaml --name A-gt-128-$USER model.num_channels=128

# 5. monitor: PD = waiting for a GPU, R = running
squeue --me
tail -f $SCRATCH/aiwcas26/logs/h2-A-mine-<jobid>.out    # submit.sh prints the exact path
#    about 4 min of silence/imports, then a progress bar with train_multi_dataset_loss_step=...

# 6. plot your loss when the job is done (writes a PNG; open it in JupyterLab's file browser)
python scripts/plot_loss.py --job <jobid>

# 7. compare with the room: open exercises/compare.ipynb in JupyterLab and Run All
python scripts/scoreboard.py show            # or just the table, in the terminal
```

`scripts/submit.sh` refuses configs that still contain `???`. It passes the config as an absolute path,
sends the log to `$SCRATCH/aiwcas26/logs/`, and runs `scripts/train.sbatch`. When the job finishes, that
script adds one row for your run to the room's scoreboard (`/leonardo_work/tra26_aiwcas/aiwcas26_scoreboard/`).
Your outputs (checkpoints, MLflow files, Hydra logs) go to `$SCRATCH/aiwcas26/hour2/<option>/`. The graph is
built on your first run and cached in `$SCRATCH/aiwcas26/hour2/graphs/`.

Rules of the queue: at most **2 running / 4 submitted jobs per person** in `boost_qos_dbg`, and the project
budget is shared by everybody, so submit one job, look at it, then decide on the next. Do **not** run
`anemoi-training train` on the login node: processes there are killed after 10 min of CPU time.

## When something goes wrong

| you see | it means | do |
|---|---|---|
| `You must specify 'model', e.g, model=<OPTION>` | a `???` in the `defaults:` list | pick one of the listed options |
| `Missing mandatory value: task.multistep_input` | a `???` left in the file | fill it (validation reports one at a time) |
| `Extra field 'num_heads' is not allowed` (or similar) | a key the chosen component does not have (e.g. `num_heads` on a GNN) | remove it, or pick the right recipe block |
| a validation error on `training.precision` | `precision: 32` is not valid | use `32-true`, `16-mixed` or `bf16-mixed` |
| `Primary config module ... not found` | `anemoi-training train` got a relative `--config-path` | use `scripts/submit.sh` (it makes the path absolute) |
| MLflow says the file store is "in maintenance mode" | `MLFLOW_ALLOW_FILE_STORE` not set | use `scripts/submit.sh` / `train.sbatch` (they set it) |
| `AssertionError: Edge attributes must be provided` | Transformer processor with a graph that has hidden→hidden edges | use the encoder/decoder-only graph and **its own file name** (A, TODO 3) |
| you changed the `graph:` section but nothing changed | the cached graph file was reused (`graph.overwrite: False`) | give the graph a new file name (or delete the old file) |
| `AssertionError: Data indexing has invalid entries ['lsm'], not in dataset` (in the job, after startup) | option 0: a forcing or diagnostic listed in `data/zarr.yaml` is missing from your `select:`. `validate` does **not** catch this | keep the 13 forcings and `tp`, `cp` |
| `found duplicate key processor` | option A: you pasted a recipe *under* `model:` instead of replacing the section | replace the whole `model:` section with the recipe |
| `num_channels Field required` / `trainable_size Extra inputs` for `PointWise...` | option A point-wise built from the packaged `model/point_wise.yaml`, which is broken in this version | use `variants/point_wise.yaml` as is |
| an error mentioning `triton` | the GPU kernel of the graph transformer failed to compile | `graph_attention_backend: pyg` in the processor and both mappers |
| `flash_attn` not found | Transformer default attention | `attention_implementation: scaled_dot_product_attention` |
| the job sits for minutes on `torch.compile` / inductor | compilation is on | `model.compile: null` |
| `DUE TO TIME LIMIT` in the log | 10 min were not enough (startup is ~4–5 of them) | fewer `training.max_steps`, fewer members (C), or a smaller rollout |
| `CUDA out of memory` | too big (unlikely with these sizes) | fewer channels, members or rollout steps |
| validation is slow (1–3 min) | Python imports from the shared filesystem | normal; don't run several at once |
| the job stays `PD` | the debug queue is busy | wait; don't submit duplicates |

Run names: put your name in them (`--name A-gt-$USER`) so the room can tell runs apart on the scoreboard.
`diagnostics.log.mlflow.experiment_name` (`hour2-0`, `hour2-A`, `hour2-C`, `hour2-B1`) is the option label on the
scoreboard: leave it as it is.

## Files

```
exercises/
  README.md                  this menu
  compare.ipynb              room-wide comparison (reads the shared scoreboard)
  0_make_it_yours/           README, skeleton.yaml, solution.yaml
  A_processor_zoo/           README, skeleton.yaml, solution.yaml (GraphTransformer), variants/{gnn,transformer,point_wise}.yaml
  C_ensemble_crps/           README, skeleton.yaml, solution.yaml
  B1_lam/                    README, skeleton.yaml, solution.yaml, lam_graph.png/.svg (map of the LAM graph)
scripts/
  submit.sh                  submit any config:  scripts/submit.sh <config.yaml> [--name N] [overrides]
  train.sbatch               the generic 1-GPU job behind submit.sh
  plot_loss.py               loss plot of one of your runs
  scoreboard.py              add / show / init the shared scoreboard
  check_exercises.py         instructor: validate (and CPU-smoke-train) many configs in one process
  make_compare_notebook.py   instructor: regenerates compare.ipynb
  crop_cerra.py              instructor: cuts the Italy window out of the CERRA sample (B1 data)
  plot_lam_graph.py          map of a B1 graph (exercises/B1_lam/lam_graph.png)
  download_data.sh           instructor: all sample data (O96, CERRA, ...) into $WORK/a08trd25/data
```

Every config is **standalone**: it lists all its settings instead of building on
`configs/04_tiny_train.yaml`, whose hidden-mesh override would break the point-wise and LAM graphs. All of them
turn off `torch.compile`, plots and extra callbacks, log to offline MLflow, and write Hydra's run directory to
`$SCRATCH/aiwcas26/hour2/hydra/`.
