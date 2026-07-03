# Method notes

## Problem setting

Both pipelines predict gene expression from H&E histology image patches for a paired
Visium HD / Visium V2 sample. Visium HD provides high-resolution (2/8/16um) binned
expression that tiles across a lower-resolution Visium V2 spot (55um), enabling
training and area-weighted evaluation at sub-spot resolution.

## Shared components

- **Encoder**: a UNI ViT-L/16 backbone (last 4 transformer blocks fine-tuned) processes
  three context views around a bin/spot center (fine, mid, coarse crops at increasing
  physical scale), plus a small CNN processes an exact-size center crop. All four
  embeddings are concatenated and fused by a trunk MLP into a 512-d representation.
- **Two-stage training**: stage 1 pretrains on highly-variable genes only (16um bins);
  stage 2 fine-tunes on the full common gene set, optionally mixing in 8um and 2um bins
  and V2 spots as auxiliary training signal.
- **Inference**: a V2 spot's 55um circular footprint is tiled with 16um-equivalent bins;
  each bin's prediction is weighted by its fractional area overlap with the circle
  (`signalgene/inference.py`).
- **HD-registered evaluation**: the HD image is registered onto the V2 image (ORB
  features + RANSAC affine, `signalgene/registration.py`), so predictions can be checked
  against an area-weighted aggregate of the true HD bins under each V2 spot, instead of
  the coarser native V2 count.

## Where the two pipelines diverge

| | `train_pathway_guided.py` | `train_baseline_direct.py` |
|---|---|---|
| Head | trunk -> TCN signal head -> gene decoder | trunk -> gene decoder |
| Intermediate target | pathway/co-expression signal (`signalgene/signalizers.py`) | none |
| Loss | signal loss + expression loss | expression loss only |
| Purpose | pathway/co-expression-guided prediction | baseline comparison |

Data, encoder architecture, training schedule, inference tiling, and evaluation are
identical between the two runs.

## Signalizers

- `coexpression_order`: hierarchically clusters genes by co-expression correlation,
  orders them by the resulting dendrogram leaf order, and averages contiguous blocks
  into `signal_len` channels.
- `pathway_group`: averages expression within named pathway gene sets, loaded from a
  JSON file (`{"pathway_name": ["GENE1", "GENE2", ...], ...}`) passed via
  `--pathway_json`. Falls back to evenly-sized contiguous gene slices if no pathway file
  is given or too few matching pathways are found.

Both signalizers are invertible: `decode()` maps a predicted signal back to per-gene
expression. The model's gene decoder head learns to approximate this mapping directly,
so the signalizer's `decode()` is not needed at inference time.
