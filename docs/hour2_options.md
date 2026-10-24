# Hour 2: "pick your setup" menu (design proposal)

Status: design only, nothing implemented. Verified against the pinned sources
(`training-0.17.1` / `models-0.19.0` / `graphs-0.9.8` / `anemoi-datasets 0.5.47`) on 2026-10-24.
Paths below are relative to `src/anemoi-core/training/` unless they say otherwise.
`CFG/` = `src/anemoi/training/config/`, `IT/` = `tests/integration/`.

**Not measured:** every timing here is an estimate. No GPU job was run for this proposal. The
only real data point is a CPU smoke run of `configs/04_tiny_train.yaml` (`outputs/2026-10-24/12-20-45/train.log`).
In that run, about 65 s went to imports, schema, graph build and model build before step 1. On GPU
the graph is cached, but imports (~30 s) and dataset setup are still paid on every job. Before the
session, run each solution config once on a GPU. Section 6 gives a checklist.

## 1. Summary

| Rank | Option | Anemoi concept | Difficulty | Data | Supported in pinned version | Loss visibly drops in ≤5 min? | Prep effort |
|---|---|---|---|---|---|---|---|
| 1 | **A. Processor zoo** (GNN / GraphTransformer / Transformer / point-wise MLP) | model = config: same data and graph, swap the processor | Easy | O96 sample (already downloaded) | Yes. GNN and GT are CI-tested (`IT/conftest.py` `global_config` params). Transformer config exists (`CFG/model/transformer.yaml`), but this env has no flash-attn, so it needs SDPA | Yes, same as base | Low |
| 2 | **B. Italy LAM / stretched grid** (CERRA 5.5 km cut into global ERA5 O96) | `cutout`, LAM graph, output mask, boundary forcing; or stretched mesh + loss re-weighting | Medium–hard | CERRA test sample (public, ~1 GB tgz; downloaded, plus an Italy crop for B1, see below) + O96 | Yes. Both are CI-tested: `IT/config/test_lam.yaml` and `test_stretched.yaml` use exactly this data pair | Probably (few variables, small model); data I/O is the unknown | Medium: download, decide the Italy crop, dry run |
| 3 | **C. Ensemble CRPS** (AIFS-ENS style) | training method ≠ task: `EnsembleTraining` + afCRPS + noise-conditioned model | Medium | O96 sample | Yes. CI-tested: `IT/config/test_ensemble_crps.yaml` on O96 | Yes (CRPS on 4 members, small model) | Low–medium (disable torch.compile, pick the attention backend) |
| 4 | **D. Temporal interpolator** (12 h → 6 h) | tasks: `TemporalDownscaler` replaces `Forecaster` | Easy–medium | O96 sample | Task exists (`src/anemoi/training/tasks/temporal_downscaler.py`). CI only tests it on a *forecast-trajectory* dataset, not on analyses → **needs a dry run** | Yes, and it ends lower than the forecaster (useful contrast) | Low |
| 5 | **E. Diffusion (EDM)** | `TransportTraining`, probabilistic generative forecaster | Hard | O96 sample | Yes. CI-tested: `test_training_cycle_edm_transport`, `IT/config/test_transport.yaml`. New code path ("transport" refactor), docs and config disagree on key names | Noisy but trending down; needs a smoothed curve | Medium |
| 6 | **F. Multi-dataset** ERA5 + CERRA (shared processor, per-dataset encoders) | `datasets:` dict-ified config, one encoder/decoder per dataset | Hard | CERRA + O96 (as B) | Yes. CI-tested: `IT/config/test_multidatasets.yaml`, `CFG/multi.yaml`. Spatial-downscaling variant: documented idea only, untested | Probably, if CERRA is thinned | Medium (shares B's data) |
| 0 | **Warm-up / fallback: "make the base yours"** (variables, rollout, multistep) | data selection, `diagnostic`/`forcing`, bounding, rollout | Easy | O96 sample | Yes (base config) | Yes | None |

**Dropped (not on the menu):**
- **Precipitation nowcasting on the DPC radar.** `$FAST/data/arco-it-dpc-sri/italian-radar-dpc-sri.zarr` is a CF/xarray zarr: `RR` with shape (3,564,898 × 1400 × 1200), 5-min steps, NaN outside coverage. It is not in anemoi format and would need `anemoi-datasets create` with the `xarray-zarr` source (`src/anemoi-datasets/src/anemoi/datasets/create/sources/xarray_zarr.py`), cropping or thinning, and NaN handling. The `indexes/mlcast-index-*.out` logs show it is being prepared with the `mlcast` tool, most likely for another session. **Ask the organizers** before using it. If they agree, it makes a good stretch option for a later edition, not for this one.
- **Observations.** A public sample exists (`.../training/datasets/test-observations-od-ai-0001-2025-noaa-19-mhs-o96-v1-testing.zarr/`), but nothing in anemoi-core 0.17.1 uses it: no task, no test, no config. Not supported in this version.
- **Spatial downscaling as its own option.** `docs/user-guide/multi-datasets.rst` describes it ("hi-res dataset all diagnostic, low-res all forcing"), but it has no config and no test. At most it is a stretch goal inside F.

## 2. Things that apply to every option

- **Base config.** `configs/04_tiny_train.yaml` exists: O96 sample, `multi_scale` graph with hidden `TriNodes` resolution 4 (2562 nodes), GNN with 64 channels and 4 layers, 300 steps, batch 1, offline MLflow. It already has `tp`/`cp` as diagnostics, and `tp` is ReLU-bounded (the `bounding` block in `CFG/model/gnn.yaml`). So "add precipitation with bounding" is not new. It is option 0's job to explain it.
- **Do not layer option skeletons on top of `04_tiny_train.yaml`.** The base sets `graph.nodes.hidden.node_builder.resolution: 4`. Hydra merges that into any other graph group, which causes these failures:
  - `encoder_decoder_only` uses `grid:` and would get an unexpected `resolution`.
  - `point_wise` has no `hidden` node set at all.
  - `limited_area` would be silently coarsened to resolution 4 (~440 km spacing).
  - `stretched_grid` uses `lam_resolution`/`global_resolution`.

  Make each skeleton a **standalone file**: copy the `system`/`dataloader`/`diagnostics`/`training` boilerplate and keep only the option-specific part as TODOs. Alternatively, move the `graph:` and `model:` overrides out of the base into a tiny `_base` mixin.
- **torch.compile.** Every GraphTransformer/Transformer model config compiles `GraphTransformerConv` by default (`compile:` in `CFG/model/graphtransformer.yaml`, `transformer.yaml`). The ensemble configs also compile `CRPS` with `mode: max-autotune` (`CFG/model/graphtransformer_ens.yaml`). That costs startup minutes in a 10-minute job, and Inductor needs a working compiler toolchain on the compute node. Set `model.compile: null` in every skeleton; the schema allows it: `compile: Optional[list[...]] = None` in `src/anemoi-core/models/src/anemoi/models/schemas/models.py`.
- **Attention backends.**
  - `graph_attention_backend: "triton"` is the default for GT mappers and processors. Triton 3.6.0 is installed, but its JIT has not been checked on a compute node. Keep `"pyg"` as a documented fallback.
  - The Transformer processor defaults to `attention_implementation: flash_attention`, and flash-attn is **not installed**. Use `scaled_dot_product_attention`, which supports sliding windows through a mask (`src/anemoi-core/models/src/anemoi/models/layers/attention.py`).
- **Head count.** The defaults are `num_heads: 16`. With `num_channels: 64` that gives a head dim of 4. Use `num_heads: 4` in the processor and mappers.
- **Turn off unneeded callbacks.** `diagnostics: evaluation_ens` adds a `RolloutEval` callback that uses `${task.validation_rollout}` (null by default). Set `diagnostics.callbacks: []`, and also `plot.callbacks: []`, as the base already does.
- **Data window.** The O96 sample has 39 six-hourly dates (2017-01-01T06 to 01-10T18). With the base split, the CPU run log shows 29 training anchors and 6 validation anchors. So 300 steps is about 10 epochs: overfitting, which is fine for "watch it go down".
- **Logged metric names** (from `src/anemoi/training/train/methods/base.py`): `train_<lossname>_loss_step`/`_epoch` and `val_<lossname>_loss`. In the base run the name is `train_multi_dataset_loss_step`. They land in the MLflow file store under `system.output.root/logs/mlflow/<exp>/<run>/metrics/`, which a notebook can read without an MLflow server.
- **Queue.** 25 students at once means 25 GPUs in `boost_qos_dbg`, with 2 running / 4 submitted jobs per user. Booster is "often nearly full" (AGENTS.md). **Ask CINECA for a reservation.** This matters more than which options are on the menu.

## 3. The options

### A. Processor zoo: same problem, four architectures (rank 1)

- **What.** Global 6-h forecaster on O96 (as in hour 1). Each student swaps the processor: `gnn` (base), `graphtransformer`, `transformer` (sliding-window attention on the hidden mesh), or `point_wise` (a per-node MLP with no spatial interaction: a "why message passing matters" baseline).
- **Shows.** Anemoi's architecture is a config choice: encoder → processor → decoder, mapped onto graph nodes and edges. The room gets the only **directly comparable** loss curves, because data, loss and scalers are identical.
- **Support.**
  - GNN and GT are CI-tested (the `global_config` fixture parametrised on `model=gnn` / `model=graphtransformer`, `IT/conftest.py`).
  - Transformer: `CFG/model/transformer.yaml`, documented in `docs/user-guide/models.rst` ("requires flash-attention", so use SDPA here).
  - Point-wise: `CFG/point_wise.yaml`, `CFG/graph/point_wise.yaml` (no edges, `hidden_nodes_name: "data"`). Not in the integration tests → dry run.
- **Data.** The O96 sample already in `$WORK/a08trd25/data/anemoi/`. No prep.
- **Config delta (TODOs).**
  - `defaults: model: ???` with one of `gnn | graphtransformer | transformer | point_wise`. For `point_wise`, also `graph: point_wise` and **no** `graph.nodes.hidden` override.
  - `model.num_channels: 64`, `model.processor.num_layers: 4`.
  - GT/T only: `model.processor.num_heads: 4`, `model.encoders."0".mapper.num_heads: 4`, `model.decoders."0".mapper.num_heads: 4`, `model.compile: null`.
  - T only: `model.processor.attention_implementation: scaled_dot_product_attention`, `model.processor.window_size: <TODO, e.g. 256>`.
  - `diagnostics.log.mlflow.run_name: <arch>-<name>`.
- **Cost.** Similar to base (tiny model, ~40k grid points, 2.5k hidden nodes). Probably limited by data loading (each sample reads 3 × 16 MB, chunks `[1,101,1,40320]`). Expect 300 steps well inside 5 min. Raise `max_steps` to 1000 if the dry run shows headroom.
- **Risks.** Triton or SDPA backend surprises (fallback: `pyg`). A `point_wise` merge conflict if layered on the base. Point-wise loss plateaus high: that is the lesson, not a bug.
- **Wow / learning.** Medium wow, high learning value. It gives a clean room-wide plot ("loss vs step and vs wall time, by architecture") and starts a discussion of inductive bias.

### B. Italy: limited-area model or stretched grid (rank 2)

- **What.** A regional 6-h forecaster on **CERRA** (5.5 km European regional reanalysis), restricted to Italy and nested in global ERA5 O96 through `cutout`. Two variants with the same data, so half the room can do each:
  - **B1 LAM.** The model predicts only the CERRA region. Global points within `max_distance_km` (300 km) are boundary forcing. Uses `output_mask: Boolean1DMask(cutout_mask)` and the `LimitedAreaTriNodes` hidden mesh.
  - **B2 Stretched grid.** The model predicts globally on a hidden mesh refined over Italy (`StretchedTriNodes`, `lam_resolution` vs `global_resolution`). Loss is re-weighted with `weight_frac_of_total` (AIFS-style regional refinement).
- **Shows.** The local relevance (FBK, Italy) is the "wow". It covers dataset composition (`cutout`, `area`, `thinning`, `adjust`), graph recipes with masks (`cutout_mask`, `boundary_mask`), and loss scalers.
- **Support.** First-class and CI-tested with exactly this data pair:
  - B1: `CFG/lam.yaml`, `CFG/graph/limited_area.yaml`, `CFG/training/lam.yaml`, `CFG/training/scalers/lam.yaml`, `IT/config/test_lam.yaml`.
  - B2: `CFG/stretched.yaml`, `CFG/graph/stretched_grid.yaml`, `CFG/training/scalers/stretched.yaml`, `IT/config/test_stretched.yaml`.
  - Tests: `test_training_cycle_lam`, `test_training_cycle_stretched` in `IT/test_training_cycle.py`.
  - `cutout`, `area`, `thinning`: `src/anemoi-datasets/docs/using/combining.rst` and `grids.rst`.
- **Data.**
  - Public: `https://object-store.os-api.cci1.ecmwf.int/ml-tests/test-data/samples/anemoi-integration-tests/training/datasets/cerra-rr-an-oper-0001-mars-5p5km-2017-2017-6h-v3-testing.zarr.tgz`. That is 987 MB compressed (an unpacked copy is also listed under the same prefix). Licence CC-BY-4.0.
  - Grid 1069×1069 (1.14 M points), **same 39 dates as the O96 sample**: this is the only public pair with matching dates. The O48 and keisler samples are from 1979.
  - Variables: `msl, sp, t_50, t_100, r_50, r_100, sf, tp` + forcings. With `adjust: all`, the common set with ERA5 is `msl, sp, t_50, t_100` (prognostic), `tp` (diagnostic) and the cos/sin forcings.
  - **Prep (instructor, ~2–3 h).**
    1. Add the tgz to the download script. **Done** (`scripts/download_data.sh`, 1.2 GB unpacked in `$WORK/a08trd25/data/anemoi/`).
    2. Pick the crop and resolution on the login node with `open_dataset`. **Done:** `area` and `thinning` combine fine (anemoi-datasets applies `thinning` first, then `area`). `{area: [48, 5, 35, 20], thinning: 4}` gives 3812 CERRA points (22 km); with `max_distance_km: 300` the cutout keeps 136 O96 boundary points (3948 in total). For comparison, CI's `thinning: 25` gives 1849 + 505.
    3. Optionally pre-build the graph. Not needed: the B1 graph (`LimitedAreaTriNodes` r7, 410 hidden nodes) builds in ~10 s, so each student builds and caches their own.
- **Config delta (TODOs).**
  - `defaults:` `graph: limited_area` + `training: lam` + `model: graphtransformer`, or `graph: stretched_grid` + `training: stretched`.
  - `dataloader.dataset.cutout: [{dataset: <CERRA>, area/thinning: TODO}, {dataset: <O96>}]`, `adjust: all`, `min_distance_km: 0`, `max_distance_km: 300` (B1).
  - `data.datasets.data.forcing` and `normalizer` lists **reduced** to the variables CERRA has (copy from `test_lam.yaml`). This is a good TODO: students hit "variable not found" if they forget.
  - B1: `model.output_mask` (from `lam.yaml`), `graph.nodes.hidden.node_builder.resolution: <TODO 6–7>`.
  - B2: `graph.nodes.hidden.node_builder.{lam_resolution, global_resolution}: <TODO, e.g. 7 / 4>`, `training.scalers.datasets.data.node_weights.weight_frac_of_total: <TODO, e.g. 0.5>`.
  - Plus the GT tweaks from section 2.
- **Cost.**
  - Model side: small (4 prognostic variables, ~10–60 k regional + ≤40 k global points).
  - Unknown: CERRA chunks are a whole 1.14 M-point field per variable and date (`chunks [1,17,1,1142761]`). Each sample decompresses ~230 MB even after cropping. With 6–8 workers that may be 2–5 samples/s, so 300 steps in ~1–3 min. **Measure it.**
  - If too slow, the instructor pre-crops CERRA once into a small anemoi zarr. One way: `anemoi-datasets create` with the `anemoi-dataset` source and an `area`.
  - **Measured / decided (B1):** on the login node a 3-date sample of the full sample took 13–22 s (mostly slow first-touch memory on the login node, not I/O). To remove the risk, `scripts/crop_cerra.py` writes a 322 × 294 grid-index window around Italy (native 5.5 km, 96 MB). It keeps the 2-D grid, so students still choose `area`/`thinning`, and a sample reads in ~0.02 s. `anemoi-datasets create` + `area` was not used, because it would flatten the grid and `thinning` would then fail.
- **Risks.**
  - Most moving parts on the menu: two datasets, masks, a variable mismatch, graph resolution confusion with the base.
  - LAM graph building with cut-off edges on many points. torch-cluster is **not installed**, which the log warns about: graph building runs on the GPU node's CPU (8 cores). Pre-build the graphs, or cap the point count.
  - Give B to stronger students, or ship a skeleton with fewer TODOs.
- **Wow / learning.** Highest wow (Italy, 5.5 km, regional nesting = what national met services do). It shows that anemoi treats LAM and stretched grids as first-class. B1 versus B2 is a good discussion: "boundary forcing vs one global model".

### C. Ensemble forecaster with CRPS, AIFS-ENS style (rank 3)

- **What.** The same O96 forecaster, now trained as an ensemble: N members per sample from noise injected into the processor (`NoiseConditioning` + `ConditionalLayerNorm`), trained with the almost-fair CRPS.
- **Shows.** Anemoi separates the **task** (what is predicted when) from the **training method** (how it is trained); see `docs/user-guide/training-methods.rst`. Deterministic to probabilistic is a handful of config keys.
- **Support.** `CFG/ensemble_crps.yaml`, `CFG/training/ensemble.yaml` (`EnsembleTraining`, `DDPEnsGroupStrategy`, `ensemble_size_per_device`), `CFG/training/training_loss/ensemble.yaml` (`MultiscaleLossWrapper` + `CRPS`, `multiscale_config: null` = single scale, no matrices needed), and `CFG/model/graphtransformer_ens.yaml`, `transformer_ens.yaml`. CI: `test_training_cycle_ensemble` with `IT/config/test_ensemble_crps.yaml` on the same O96 sample. CI uses multiscale smoothing matrices from `.../training/assets/o96_smoothing_o96.npz` (3.8 MB, public); we do not need them.
- **Data.** O96 sample (deterministic analyses; the members come from model noise, which is the AIFS-CRPS setup). The tiny `aifs-ea-an-enda-...-20p0` sample (3-member EDA, 9×18 grid) is already downloaded, but it is far too coarse to train on. Use it only to show "ensemble datasets exist" in the dataset notebook.
- **Config delta (TODOs).**
  - `defaults: model: graphtransformer_ens` (keeps the base `multi_scale` graph and avoids flash-attn), `training: ensemble`, `diagnostics: evaluation_ens`.
  - `training.ensemble_size_per_device: <TODO 2–4>`, `system.hardware.num_gpus_per_ensemble: 1`.
  - `model.compile: null` (**essential**: default is `max-autotune` on CRPS).
  - `model.noise_injector.noise_channels_dim` (optional TODO), `training.training_loss.datasets.data.per_scale_loss.alpha` (optional: fair vs standard CRPS).
  - `diagnostics.callbacks: []`, plus the GT tweaks.
- **Cost.** About 2–4× the base per step (N members through the processor). With 64 channels, 4 layers and N = 4, still likely < 1 s/step, so 200–300 steps fit. CRPS drops quickly from the untrained spread.
- **Risks.** The torch.compile and triton issues above. `RolloutEval` with null rollout. The loss value is not comparable with MSE runs, so the comparison step must handle it (section 5). Moderate robustness once the solution is dry-run.
- **Wow / learning.** High: "AIFS-ENS is the same codebase, flip the method". It suggests a nice extra plot: ensemble spread per member from the checkpoint (optional).

### D. Temporal interpolator, 12 h → 6 h (rank 4)

- **What.** Given states at t and t+12 h, predict t+6 h, with `TemporalDownscaler` replacing `Forecaster`.
- **Shows.** The task abstraction: `docs/user-guide/tasks.rst` lists four built-in tasks (Forecaster, OffsetForecaster [experimental], TemporalDownscaler, Autoencoder). The same model learns a different problem.
- **Support.**
  - Code: `src/anemoi/training/tasks/temporal_downscaler.py`. Offsets `[0, 12H]` → `[6H]` are checked for compatibility with the 6-h dataset frequency in `src/anemoi/training/data/relative_time_indices.py`, so the dataset gets indices 0/1/2.
  - Configs: `CFG/task/temporal_downscaler.yaml`, `CFG/temporal_downscaler.yaml`.
  - CI (`test_training_cycle_temporal_downscaler`) only runs it on the **forecast trajectory** sample `aifs-od-fc-oper-0001-mars-o48-2017-2017-6h-1h-v5-testing.zarr` (634 MB, public), using the `forecast_native_grid` dataloader (`step_start`/`step_end`). On analysis data with the plain `native_grid` dataloader it is **untested**. It should work (generic offsets), but needs a dry run.
- **Data.** O96 sample. Optional upgrade: the public O48 forecast sample with 1-h steps, for a "6 h → 1 h" variant matching CI exactly.
- **Config delta (TODOs).**
  - `defaults: task: temporal_downscaler`; keep `dataloader: native_grid`. Do **not** take `forecast_native_grid` from `CFG/temporal_downscaler.yaml`.
  - `task.input_timestep: 12H`, `task.output_timestep: 6H`, optional `output_left_boundary`/`output_right_boundary`.
  - `training.scalers.datasets.data.{stdev,var}_tendency.timestep: ${task.output_timestep}` (from `CFG/temporal_downscaler.yaml`).
  - Optional `override training/training_loss: single_MSE_aggregation`.
  - The model can stay `gnn` (SingleTraining supports it per `training-methods.rst`).
- **Cost.** Same as base (3 dates per sample). The loss should fall faster and lower than the forecaster: interpolation is easier. That makes a good A/D talking point.
- **Risks.** The untested analysis-data path. Few valid anchors (~27). Task-specific plot adapters if plot callbacks get switched on.
- **Wow / learning.** Medium wow, good conceptual value. Cheap to implement once the dry run passes.

### E. Diffusion forecaster, EDM (rank 5)

- **What.** A probabilistic forecaster trained by denoising (EDM / Karras noise schedule): `TransportTraining` with `objective: edm_diffusion`.
- **Shows.** Generative ML in the same framework; the AIFS diffusion line.
- **Support.** `CFG/transport_edm_diffusion.yaml`, `CFG/training/transport.yaml`, `CFG/training/edm_diffusion.yaml`, `CFG/model/graphtransformer_transport_edm.yaml`. CI: `test_training_cycle_edm_transport` on the O96 sample (`IT/config/test_transport.yaml`), plus a stochastic-interpolant variant (`transport_stochastic_interpolant.yaml`, config-validation test only).
  - Caveats: GNN is not supported (`docs/user-guide/training-methods.rst`). The docs say `training.transport_objective`, but the shipped config uses `training.transport.objective`; this code is recent and still moving.
- **Data.** O96 sample.
- **Config delta (TODOs).**
  - `defaults: model: graphtransformer_transport_edm`, `training: edm_diffusion`.
  - `model.model.transport.{sigma_max, sigma_min, sigma_data}` (optional TODOs).
  - `training.precision` (default `bf16-mixed`), `model.compile: null`, `model.bounding.datasets.data: []` (CI disables bounding for transport), plus the GT tweaks.
- **Cost.** One forward pass per step at a random noise level, so about the cost of GT. Sampling (50 Heun steps) is not needed for training. The loss is very noisy (random σ per step): plot an EMA-smoothed curve. The 5-min trend is likely visible but shallow.
- **Risks.** Least mature path. The loss curve is the least convincing to a student ("is it learning?"). Showing samples would need inference, which is out of scope.
- **Wow / learning.** High wow on paper. For a 5-minute run, offer it only to advanced students.

### F. Multi-dataset: ERA5 + CERRA with a shared processor (rank 6)

- **What.** Two datasets, each with its own encoder and decoder, one shared latent processor (`AnemoiModelEncProcDec` with `encoders."0"`/`"1"`). Stretch variant: spatial downscaling (ERA5 all forcing → CERRA all diagnostic).
- **Shows.** The newest anemoi capability: dict-ified `datasets:` everywhere (`docs/user-guide/multi-datasets.rst`).
- **Support.** `CFG/multi.yaml`, `CFG/data/multi.yaml`, `CFG/dataloader/multi.yaml`, `CFG/graph/multi.yaml`, `CFG/model/graphtransformer_multi.yaml`. CI: `test_training_cycle_multidatasets` with `IT/config/test_multidatasets.yaml` (same O96 + CERRA pair). The downscaling variant has no config or test.
- **Data.** As B. CERRA must be thinned in `dataloader.*.datasets.cerra.dataset_config` (CI uses it at full 1.14 M points with 16 channels).
- **Config delta (TODOs).** Large. Every `data`-keyed block becomes `era5:`/`cerra:` (forcing lists, normalisers, scalers, losses, metrics, plots). Too many TODOs for 15 minutes, unless the skeleton pre-fills most of it and leaves a few TODOs (CERRA forcings and diagnostics, thinning, node names).
- **Cost.** As B, plus a second encoder/decoder.
- **Risks.** Highest config complexity, same I/O unknown as B.
- **Wow / learning.** Conceptually strong. Practically it is better shown by an instructor demo, or offered as the "I finished early" option.

### 0. Warm-up / fallback: "make the base yours"

- **What.** The base forecaster with student choices:
  - variable selection (`dataloader.dataset: {dataset: ..., select: [...]}` or `drop`);
  - moving `tp` between `diagnostic` and prognostic;
  - removing the `tp` bounding to see the effect;
  - `task.multistep_input`, a rollout schedule (`task.rollout.maximum: 2–3`), `training.optimization.lr`.
- **Why.** Zero risk. It is for students whose chosen option fails or who want to stay simple, and it gives a clean, comparable MSE (but compare only same-variable runs).

## 4. Presenting the menu

- A single markdown page, `docs/hour2_menu.md`, or the top cell of a `notebooks/05_choose_your_setup.ipynb`. It holds the summary table above, trimmed to: option, one-line pitch, concept, difficulty (★ / ★★ / ★★★), expected run time, and the file names `configs/hour2/<opt>_skeleton.yaml` and `<opt>_solution.yaml`.
- Show one picture per option (for example the LAM/stretched graph plot, ensemble members, the zoo diagram) so people choose by concept, not by name.
- **Steer the distribution.** The room should cover several setups, which also spreads the risk. For example, ask by show of hands or a shared sheet: A-zoo ~8 (2 per architecture), B ~6 (3 LAM, 3 stretched), C ~5, D ~4, E/F ~2.
- Each skeleton has numbered `# TODO(n):` comments, each with a one-line hint and a link to the relevant hour-1 notebook cell. The steps are the same for every option: `validate` → `sbatch` → plot.

## 5. Common comparison step

Losses differ across options (MSE vs afCRPS vs EDM-weighted MSE, different variables and domains), so the raw numbers are not comparable. Instead:

1. **A shared scoreboard.** At the end of the job (or from the notebook), each run appends one row to a group-writable CSV in `$WORK` (for example `$WORK/aiwcas26_scoreboard.csv`). Columns: `student, option, variant, n_params, s_per_step, peak_gpu_mem, steps, loss_first, loss_last, val_loss_last`. All of these come from the MLflow file store and the Lightning logs.
2. **A normalised loss plot for the whole room.** Plot `train_loss_step / train_loss_step[first 10 steps mean]` against step (and against wall time), coloured by option, with an EMA smoothing slider. This answers "did everyone learn, and how fast", whatever the loss.
3. **Like-for-like panels.** Raw `val_*` curves only within comparable groups:
   - A (all architectures) + 0 + D share the O96 MSE family; D only against the forecaster, with the caveat that interpolation is easier.
   - B1 vs B2: both log `mse_inside_lam` (`CFG/training/lam.yaml`, `stretched.yaml`), so they are comparable on the Italy region.
   - C: CRPS vs the deterministic MSE. Optionally make it comparable with a small add-on: CRPS of a 1-member ensemble equals MAE, which an instructor can pre-compute.
4. **Cost per idea.** A bar chart of parameters and s/step per option. It shows that LAM, ensemble and diffusion have very different compute profiles.

All of it lives in one notebook (`notebooks/06_compare.ipynb`) that globs `$SCRATCH`/`$WORK` run directories (read-only) or reads the CSV. No MLflow server is needed (offline file store).

## 6. Implementation order if time is short

1. **A (zoo) + 0 (fallback).** O96 data only, one solution per architecture (4 short yaml files), almost no new risk. On its own it already gives the "compare across the room" moment.
2. **C (ensemble CRPS).** O96 data only, CI-tested, a strong "anemoi does many problems" message.
3. **B1 (Italy LAM).** The local wow. Needs the CERRA download and a dry run. Add B2 (stretched) once B1 works: it is mostly the same file.
4. **D (interpolator).** Cheap if its dry run passes.
5. E and F only if there is spare instructor time. Otherwise show them as "also possible" slides.

Minimal viable menu: **A, C, B1** (+0). Together they cover architecture, probabilistic training and regional modelling, with only one extra dataset to download.

**Per-option dry-run checklist (instructor, 1 GPU, debug QOS):**
1. `config validate` passes.
2. Time to first step < 3 min.
3. 300 steps in < 5 min.
4. The loss ratio last/first is clearly < 1.
5. No triton or torch.compile JIT on the compute node (or record how long it takes).
6. Note the peak memory.
7. Record s/step in the menu table.

## 7. Main uncertainties

- No GPU timings yet. Data-loader throughput for CERRA (B, F) is the biggest unknown.
- ~~Whether `area` + `thinning` on CERRA's 2-D Lambert grid behaves (B).~~ It does (B1 uses `area: [48, 5, 35, 20], thinning: 4`). CERRA I/O is handled by a pre-cropped window (`scripts/crop_cerra.py`).
- `TemporalDownscaler` on an analysis dataset with the plain dataloader is not covered by CI (D).
- Triton graph-attention JIT and Inductor on Leonardo compute nodes (A-GT, C, E, F). Mitigation: `compile: null`, `graph_attention_backend: pyg`.
- GPU availability in `boost_qos_dbg` for ~25 simultaneous jobs: ask for a reservation.
- Radar data ownership: ask the organizers.
