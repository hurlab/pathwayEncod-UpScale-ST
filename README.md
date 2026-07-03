# Pathway-Guided Gene Expression Prediction from Histology

Predicting spatial gene expression directly from H&E histology images, using a
paired Visium HD (high-resolution binned) / Visium V2 (spot-resolution) sample for
training and evaluation.

This repo contains two pipelines that share the same image encoder, training
schedule, and evaluation procedure:

- **`scripts/train_pathway_guided.py`** — predicts a biologically structured
  intermediate signal from the image first (via co-expression clustering or pathway gene sets),
  then decodes it into per-gene expression.
- **`scripts/train_baseline_direct.py`** — predicts gene expression directly from the
  image trunk, with no intermediate signal. Baseline for comparison.

See [`docs/METHOD.md`](docs/METHOD.md) for the shared architecture, training/inference
procedure, and how the two pipelines differ.

## Repository layout

```
configs/
  paths.example.yaml     # template for your local dataset paths (copy -> paths.yaml)
src/signalgene/
  config.py              # dataset path resolution
  bundle.py              # loads images/expression/coords for one sample
  io.py                  # expression matrix / coordinate / whole-slide image loaders
  patches.py             # multi-scale patch extraction, area-overlap weighting
  genes.py               # highly-variable gene selection
  signalizers.py         # coexpression_order / pathway_group signal encoders
  models.py              # shared encoder + signal model + direct baseline model
  datasets.py            # patch dataset (signal target optional)
  engine.py              # train / eval loop
  inference.py           # weighted area-overlap V2 spot inference
  registration.py        # HD<->V2 image registration + HD-registered comparison
  plotting.py            # training curve plots
  constants.py           # shared constants and default hyperparameters
  metrics.py             # Pearson / Spearman / RMSE helpers
  utils.py               # seeding, checkpoint loading, DataParallel wrapping
scripts/
  train_pathway_guided.py
  train_baseline_direct.py
docs/
  METHOD.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA GPU for practical training times (multi-GPU via `--gpus` uses
`DataParallel`); CPU works but will be very slow.

## Data

Each pipeline expects one paired sample:

- **Visium HD**: a whole-slide H&E image plus 2um/8um/16um square-binned
  `filtered_feature_bc_matrix.h5` outputs (standard Space Ranger HD output layout).
- **Visium V2**: a whole-slide H&E image, `tissue_positions.csv`, and a
  `filtered_feature_bc_matrix` (h5 or mtx directory) from the matched tissue.

Copy `configs/paths.example.yaml` to `configs/paths.yaml` and fill in the paths to
your own data — `paths.yaml` is gitignored so your local filesystem layout never ends
up in version control.

```bash
cp configs/paths.example.yaml configs/paths.yaml
# edit configs/paths.yaml
```

## Usage

Baseline (direct, no signal):

```bash
python scripts/train_baseline_direct.py --config configs/paths.yaml --use_hd8 --use_hd2 --gpus 0
```

Pathway-guided, single method:

```bash
python scripts/train_pathway_guided.py --config configs/paths.yaml \
  --method coexpression_order --use_hd8 --use_hd2 --gpus 0 2 3

python scripts/train_pathway_guided.py --config configs/paths.yaml \
  --method pathway_group --pathway_json pathways.json --use_hd8 --use_hd2
```

Run every signalizer method and write a combined comparison CSV:

```bash
python scripts/train_pathway_guided.py --config configs/paths.yaml \
  --run_all_methods --pathway_json pathways.json --use_hd8 --use_hd2 --gpus 0 2 3
```

`--pathway_json` expects `{"pathway_name": ["GENE1", "GENE2", ...], ...}`. Without it
(or with too few matching pathways), `pathway_group` falls back to evenly-sized
contiguous gene slices.

### Key arguments (both scripts)

| Flag | Meaning |
|---|---|
| `--config` | Path to the dataset paths YAML (default `configs/paths.yaml`) |
| `--out_dir` | Overrides `output_dir` from the config file |
| `--use_hd8` / `--use_hd2` | Add 8um / 2um HD bins as auxiliary training data |
| `--use_v2_train` | Also train on V2 spots directly |
| `--gpus 0 2 3` | GPU IDs; first is primary, multiple enables `DataParallel` |
| `--n_v2_infer` | Number of V2 spots to run inference/evaluation on |

Run `--help` on either script for the full list (epochs, batch size, learning rates,
gene/HVG counts, per-resolution sample caps).

## Outputs

Each run writes to its output directory:

- `stage{1,2}_history.csv`, `stage{1,2}_{loss,corr}.png` — training curves
- `stage{1,2}_best*.pt` — checkpoints (encoder + head weights, gene list, and for the
  pathway-guided pipeline, the fitted signalizer state)
- `v2_spot_metrics.csv`, `v2_gene_metrics.csv`, `v2_predictions.csv` — V2 spot-level
  predictions and spot-/gene-level Pearson, Spearman, RMSE
- `hd_comparison.csv`, `registration_info.json` — HD-registered comparison against
  registered HD ground truth
- `summary.json` — a single-file rollup of all the above metrics

The pathway-guided pipeline nests these under a per-method subdirectory
(`<out_dir>/<method>/...`) and adds `all_methods_comparison.csv` when run with
`--run_all_methods`.

## License

MIT — see [LICENSE](LICENSE).
