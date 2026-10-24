# Option 0 · Make the base yours ★

**The hour-1 forecaster, with your choices.** It is the same small GNN as `configs/04_tiny_train.yaml`,
written as a standalone file. You decide **what** it predicts, **what it cares about** in the loss, **how
much history** it sees and **how far** it is trained to forecast on its own.

This is also the **fallback**: if your chosen option misbehaves, copy `solution.yaml`, change the run
name and submit it.

## Why

Most of the work on a real forecasting model is not the architecture. It is these knobs:

| knob | config key | what it changes |
|---|---|---|
| variables | `dataloader.*.datasets.data.dataset_config.select` (or `drop`) | the inputs and outputs. Each variable is **forcing** (input only), **diagnostic** (output only, e.g. `tp`) or **prognostic** (both), set in `data.datasets.data.forcing` / `diagnostic` |
| loss weights | `training.scalers.datasets.data.general_variable.weights` | how much each variable counts in the MSE. AIFS weights `z` 12×, `w` 0.001× |
| input history | `task.multistep_input` | how many past states the model sees (t, t−6h, …) |
| rollout | `task.rollout.{start,maximum}` | how many 6-h steps the model runs on its **own** output inside one training step. This trains stability; the cost grows linearly |

## What to do

1. `cp skeleton.yaml mine.yaml` and fill the `???`:
   * **TODO(1)** `select:` your variable list. Keep the 13 forcings and `tp`, `cp`: the packaged
     `data/zarr.yaml` names them, and they must exist. Example: surface only, or surface + 500 and 850 hPa.
   * **TODO(2)** a weight for `2t` in the loss (the default is 1). You can add more lines, e.g. `msl: 2`.
     Weights apply to a variable (`t_850`) or to a whole parameter (`t` = every level).
   * **TODO(3)** `multistep_input`: 1, 2 or 3.
   * **TODO(4)** rollout: the same value for `start` and `maximum` (1 = no rollout, 2–3 is fine in 10 min).
     Every extra step removes one of the 29 training samples.
   * **TODO(5)** a run name that says what you chose, e.g. `0-sfc-ms1-r2-<you>`.
2. Validate, submit, plot (see `../README.md`).

## Things to try

* Make `tp` prognostic: remove it from `data.datasets.data.diagnostic`, so it also becomes an input.
  The `ReluBounding` on `tp` in `model.bounding` keeps it ≥ 0. What happens if you remove the bounding
  (`model.bounding.datasets.data: []`)?
* Rollout 3 vs 1 with everything else equal: is the loss higher? (It averages harder, longer forecasts.)
* Drop the pressure levels entirely: how much faster per step is a 20-variable model?

## Expected outcome

The loss decreases clearly within 500 steps. The solution (33 variables, `2t`×5, multistep 1, rollout 2)
is GPU-tested; see `../README.md` for its numbers. **Raw losses are only comparable between runs with the
same variables, rollout and weights.** The scoreboard puts those in one `compare_group`, and
`compare.ipynb` normalises the curves for everything else.
