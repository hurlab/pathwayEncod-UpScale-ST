"""Training curve plots."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_training_plots(stage1_hist, stage2_hist, out_dir: str, tag: str = ""):
    suffix = f"_{tag}" if tag else ""
    title_suffix = f" - {tag}" if tag else ""
    for hist, stage in [(stage1_hist, "stage1"), (stage2_hist, "stage2")]:
        if not len(hist):
            continue
        plt.figure(figsize=(7, 5))
        plt.plot(hist["epoch"], hist["train_loss"], label="train")
        plt.plot(hist["epoch"], hist["val_loss"], label="val")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title(f"{stage} loss{title_suffix}"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{stage}_loss{suffix}.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(hist["epoch"], hist["val_gene_corr"], label="val_gene_corr")
        plt.xlabel("Epoch"); plt.ylabel("Mean gene log1p Pearson")
        plt.title(f"{stage} gene correlation{title_suffix}"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{stage}_corr{suffix}.png"), dpi=150)
        plt.close()
