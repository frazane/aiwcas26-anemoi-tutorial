# Option C · Ensemble forecaster with CRPS, AIFS-ENS style ★★

**From one forecast to a probabilistic one.** The same O96 forecaster now produces several forecasts
(**members**) per sample. The members differ only in random noise injected into the processor: a
`NoiseConditioning` module turns a noise vector into the scale and shift of every `ConditionalLayerNorm`.
They are trained together with the **almost-fair CRPS** (afCRPS), the loss of ECMWF's AIFS-ENS.

## Why

* **Task ≠ training method.** The *task* (`task: forecaster`: two input states → the next state) is
  unchanged. Only the *training method* changes: `training: ensemble` brings in `EnsembleTraining`, an
  ensemble-aware strategy and a CRPS loss. A deterministic → probabilistic switch is a handful of config
  keys.
* **Why CRPS?** MSE rewards the ensemble *mean*, which is smooth and blurry. CRPS rewards a sharp,
  well-spread ensemble: CRPS = E|X − y| − ½ E|X − X′| (accuracy minus spread). With few members the plain
  estimate favours too little spread. The *fair* CRPS (α = 1) corrects that bias. AIFS-ENS uses α = 0.95
  ("almost fair") because the fully fair version can let members collapse onto each other.

## What to do

1. `cp skeleton.yaml mine.yaml` and fill the `???`:
   * **TODO(1)** `model: graphtransformer_ens`, the noise-conditioned GraphTransformer.
     `transformer_ens` exists too, but it needs flash-attn.
   * **TODO(2)** `training: ensemble`.
   * **TODO(3)** `training.ensemble_size_per_device`: members per sample (≥ 2; 4 is tested).
   * **TODO(4)** `system.hardware.num_gpus_per_ensemble: 1`. Members can be spread over GPUs; we have one.
   * **TODO(5)** `model.compile: null`. The ensemble model's default compiles the CRPS loss with
     `mode: max-autotune`, which costs minutes of a 10-minute job.
   * **TODO(6)** `alpha` of the CRPS: 1.0 fair, 0.0 standard, 0.95 AIFS-ENS.
   * **TODO(7)** a run name, e.g. `C-ens4-a095-<you>`.
2. Validate, submit, plot (see `../README.md`).

`diagnostics: evaluation_ens` adds a `RolloutEval` callback that needs `task.validation_rollout`. The
skeleton already switches it off with `diagnostics.callbacks: []`, and the ensemble plots with
`diagnostics.plot.callbacks: []`.

## Things to try

* `alpha` 1.0 vs 0.0 vs 0.95, everything else equal. The loss values differ by definition: compare the
  normalised curves.
* 2 vs 4 vs 8 members: the scoreboard shows what each costs (`s/step`, GPU memory).
* `model.noise_injector.noise_channels_dim`: a smaller or larger noise vector.

## Expected outcome

The afCRPS drops clearly within the job: members start with a random spread and learn to agree with the
analysis. The value is **not** comparable with the MSE of options 0/A, so use the normalised plot in
`compare.ipynb`. At this tiny size a training step costs about the same as option A's deterministic GraphTransformer
(9.2 vs 9.5 it/s), but it needs twice the GPU memory and validation is slower. At production size every
member is a full pass through the model, so the cost grows with the number of members. GPU-tested numbers are in `../README.md`.
