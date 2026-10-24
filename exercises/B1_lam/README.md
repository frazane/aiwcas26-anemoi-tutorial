# Option B1 · Italy limited-area model (CERRA cutout): PENDING DATA

**Not available yet.** This option needs the CERRA 5.5 km sample, which is not on Leonardo yet: data access
is pending. Do not try to download it during the session. Pick option 0, A or C instead.

## What it will be

A regional 6-hour forecaster on **CERRA** (5.5 km European regional reanalysis) restricted to Italy, nested
in the global ERA5 O96 sample with anemoi-datasets' `cutout`. The model predicts only inside the CERRA
region. Global points within `max_distance_km` act as boundary forcing.

Anemoi concepts: dataset composition (`cutout`, `area`, `thinning`, `adjust`), a limited-area graph
(`graph: limited_area`, `LimitedAreaTriNodes`, `cutout_mask`), `model.output_mask`, LAM loss scalers and
the `mse_inside_lam` metric.

## What is needed to finish it (instructor)

1. **Data.** Fetch the public test sample
   `cerra-rr-an-oper-0001-mars-5p5km-2017-2017-6h-v3-testing.zarr` (987 MB tgz; same 39 dates as the O96
   sample) from the anemoi integration-test object store, on a login node. Add it to `scripts/download_data.sh`.
2. **Crop.** On the login node, check with `open_dataset` that
   `{dataset: <cerra>, area: [48, 5, 35, 20], thinning: N}` works on CERRA's 2-D Lambert grid.
   The fallback is CI's `thinning: 25` over the full domain (~43×43 points).
   Measure the samples/s of the data loader: each CERRA chunk is a whole 1.14 M-point field.
   If loading is too slow, pre-crop CERRA once into a small zarr.
3. **Configs.** `skeleton.yaml` and `solution.yaml` here, standalone like the other options. Start from
   `src/anemoi-core/training/tests/integration/config/test_lam.yaml` and `.../config/lam.yaml`:
   * `defaults`: `graph: limited_area`, `training: lam`, `model: graphtransformer`, plus the usual
     `model.compile: null`, `num_heads: 4`, and graph-attention backend tweaks;
   * `dataloader.dataset: {cutout: [{dataset: <cerra crop>}, {dataset: <o96>}], adjust: all, min_distance_km: 0}`;
   * `data.datasets.data.forcing` and the normaliser lists reduced to the variables CERRA has (copy from
     `test_lam.yaml`). This makes a good TODO: "variable not found" if forgotten;
   * `graph.nodes.hidden.node_builder.resolution`: a TODO (6–7). Pre-build the graph, because torch-cluster
     is not installed and cut-off edges on many points are slow;
   * `diagnostics.log.mlflow.experiment_name: hour2-B1` (the scoreboard uses it as the option label).
4. **Dry run** with `scripts/submit.sh exercises/B1_lam/solution.yaml` and record the timings in
   `exercises/README.md`. `scripts/scoreboard.py` needs no change: B1 runs get their own comparison group,
   and `compare.ipynb` already has a colour for B1.
