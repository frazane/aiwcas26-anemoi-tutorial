# Option A · Processor zoo ★

**Same problem, four architectures.** Everybody in option A trains the same global 6-hour forecaster on the
O96 sample, with the same encoder input, hidden mesh, loss, steps and learning rate. Only the **processor**
changes: the part of encoder → processor → decoder that works on the hidden mesh.

| processor | `model:` | what the hidden nodes do | graph |
|---|---|---|---|
| GNN | `gnn` | message passing along mesh edges (sum of learned messages from neighbours) | `multi_scale` |
| GraphTransformer | `graphtransformer` | attention, but only along mesh edges | `multi_scale` |
| Transformer | `transformer` | global self-attention: every hidden node sees all 2562 others, no edges | `multi_scale` nodes, encoder/decoder edges only |
| point-wise MLP | `point_wise` | nothing spatial: an MLP applied to every grid point independently | `point_wise` (no edges) |

## Why

In anemoi the architecture is a **config choice**. `model: gnn` → `model: graphtransformer` swaps the
`_target_` classes that Hydra instantiates, and the rest of the pipeline stays the same: data, normalisers,
graph nodes, loss, training loop. Because data, loss and scalers are identical, option A gives the only
**directly comparable** loss curves in the room. The comparison is about inductive bias: what does the model
assume about how information moves around the globe?

## What to do

1. `cp skeleton.yaml mine.yaml` and fill the `???`:
   * **TODO(1)** `model:` one of `gnn | graphtransformer | transformer`. Agree in the room so every
     architecture is covered. **Point-wise is the exception:** the packaged `model/point_wise.yaml` is broken
     in this anemoi version, so point-wise people copy `variants/point_wise.yaml`, change its `run_name` and
     submit it. Read its model section: it is the whole architecture written out.
   * **TODO(2)** the graph: all three keep `graph: multi_scale` (same nodes). The transformer replaces the
     edge list: no hidden→hidden edges.
   * **TODO(3)** the graph file name, one per recipe (listed in the comment). The graph is built on the
     first run and then **reused**, so a file with the wrong edges gives confusing errors.
   * **TODO(4)** the processor-specific model section. GNN: keep it as it is. GraphTransformer or
     Transformer: **replace** the whole `model:` section (and, for the Transformer, the `graph:` section) with
     the block from `RECIPES` at the end of the skeleton. Pasting *under* `model:` gives
     `found duplicate key processor`. The schema rejects keys a processor does not know
     (`Extra field 'num_heads' is not allowed`).
   * **TODO(5)** a run name, e.g. `A-gt-<you>`.
2. Validate, submit, plot (see `../README.md`).

## Things to try once it runs

* Same architecture, more capacity: `--name A-gt-128 model.num_channels=128`. Is it better per step?
  Per minute?
* Transformer: `model.processor.window_size=512` makes the attention local (along the node order) instead
  of global.
* GraphTransformer: `graph_attention_backend: pyg` instead of `triton`. How much slower is it?

## Expected outcome

All four learn within 500 steps. The point-wise MLP should plateau visibly higher: without any spatial
interaction the model can only learn a per-point tendency. GNN, GraphTransformer and Transformer end close
together on this tiny problem (29 training samples). Their cost per step differs: compare `s/step` on the
scoreboard. Reference GPU dry runs are in `../README.md`.

| file | what |
|---|---|
| `skeleton.yaml` | to fill |
| `solution.yaml` | GraphTransformer answer (GPU-tested) |
| `variants/gnn.yaml`, `variants/transformer.yaml`, `variants/point_wise.yaml` | the other answers (all validated; transformer and point_wise also CPU-trained; none GPU-tested yet; gnn = the GPU-tested hour-1 model) |
