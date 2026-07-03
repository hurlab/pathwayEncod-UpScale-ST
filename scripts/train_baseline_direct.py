#!/usr/bin/env python3
"""Baseline pipeline: direct image -> gene expression prediction, no signal bottleneck.

Same encoder, training schedule, inference, and evaluation as the pathway-guided
pipeline (scripts/train_pathway_guided.py), but the gene decoder attaches straight
to the trunk output instead of going through a signal head.

Example:
    python scripts/train_baseline_direct.py --config configs/paths.yaml --use_hd8 --use_hd2 --gpus 0
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from signalgene.bundle import load_sample_bundle
from signalgene.config import load_dataset_paths
from signalgene.constants import DEFAULTS, HD_CENTER_BOX, SEED, V2_CENTER_BOX
from signalgene.datasets import PatchDataset
from signalgene.engine import evaluate, train_one_epoch
from signalgene.genes import select_hvg
from signalgene.inference import run_v2_inference
from signalgene.models import ImageToGeneModel
from signalgene.plotting import save_training_plots
from signalgene.registration import run_hd_bin_comparison
from signalgene.utils import safe_torch_load, seed_everything, unwrap, wrap_model
from scipy.sparse import issparse


def build_optimizer(model, lr_head, lr_backbone, weight_decay):
    base = unwrap(model)
    backbone_params = [p for n, p in base.named_parameters() if p.requires_grad and n.startswith("encoder.backbone")]
    head_params = [p for n, p in base.named_parameters() if p.requires_grad and not n.startswith("encoder.backbone")]
    return torch.optim.AdamW(
        [{"params": head_params, "lr": lr_head}, {"params": backbone_params, "lr": lr_backbone}],
        weight_decay=weight_decay,
    )


def run(args, shared, out_dir, gpu_ids, device):
    os.makedirs(out_dir, exist_ok=True)
    hd_mem, v2_mem = shared["hd_mem"], shared["v2_mem"]
    hd2_ad, hd8_ad, hd16_ad, v2_ad = shared["hd2_ad"], shared["hd8_ad"], shared["hd16_ad"], shared["v2_ad"]
    hd2_pos, hd8_pos, hd16_pos, v2_pos = shared["hd2_pos"], shared["hd8_pos"], shared["hd16_pos"], shared["v2_pos"]
    common_genes = shared["common_genes"]

    X16_all = hd16_ad[:, common_genes].X
    X16_all = X16_all.toarray() if issparse(X16_all) else np.asarray(X16_all)
    _, hvg_genes = select_hvg(X16_all, common_genes, n_top=min(args.hvg, len(common_genes)))

    ds16_s1 = PatchDataset(hd_mem, hd16_pos, hd16_ad, hvg_genes, 16, HD_CENTER_BOX[16], max_samples=args.max_hd16)
    ds16_s2 = PatchDataset(hd_mem, hd16_pos, hd16_ad, common_genes, 16, HD_CENTER_BOX[16], max_samples=args.max_hd16)
    ds8_s2 = PatchDataset(hd_mem, hd8_pos, hd8_ad, common_genes, 8, HD_CENTER_BOX[8], max_samples=args.max_hd8) if args.use_hd8 else None
    ds2_s2 = PatchDataset(hd_mem, hd2_pos, hd2_ad, common_genes, 2, HD_CENTER_BOX[2], max_samples=args.max_hd2) if args.use_hd2 else None
    dsv2_s2 = PatchDataset(v2_mem, v2_pos, v2_ad, common_genes, 55, V2_CENTER_BOX, max_samples=args.max_v2) if args.use_v2_train else None

    def split(ds):
        n = len(ds)
        n_train = int(0.8 * n)
        return random_split(ds, [n_train, n - n_train], generator=torch.Generator().manual_seed(SEED))

    tr16_s1, va16_s1 = split(ds16_s1)
    tr16_s2, va16_s2 = split(ds16_s2)

    eff_s1 = args.batch_stage1 * max(1, len(gpu_ids))
    eff_s2 = args.batch_stage2 * max(1, len(gpu_ids))

    def loader(ds, batch, shuffle):
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True)

    tr16_s1_loader, va16_s1_loader = loader(tr16_s1, eff_s1, True), loader(va16_s1, eff_s1, False)
    tr16_s2_loader, va16_s2_loader = loader(tr16_s2, eff_s2, True), loader(va16_s2, eff_s2, False)
    aux_loaders = {}
    if ds8_s2 is not None:
        aux_loaders["hd8"] = loader(ds8_s2, eff_s2, True)
    if ds2_s2 is not None:
        aux_loaders["hd2"] = loader(ds2_s2, eff_s2, True)
    if dsv2_s2 is not None:
        aux_loaders["v2"] = loader(dsv2_s2, eff_s2, True)
    aux_weights = {"hd8": 0.35, "hd2": 0.20, "v2": 0.35}

    model_s1 = wrap_model(ImageToGeneModel(args.uni_weights, len(hvg_genes)), gpu_ids, device)
    model_s2 = wrap_model(ImageToGeneModel(args.uni_weights, len(common_genes)), gpu_ids, device)
    if len(gpu_ids) > 1:
        print(f"  DataParallel on GPUs: {gpu_ids}")

    opt_s1 = build_optimizer(model_s1, args.lr_head, args.lr_backbone, args.weight_decay)
    opt_s2 = build_optimizer(model_s2, args.lr_head, args.lr_backbone, args.weight_decay)

    # Stage 1: pretrain on HVGs only.
    hist1, best1 = [], float("inf")
    ckpt1 = os.path.join(out_dir, "stage1_best.pt")
    ep = 0
    try:
        for ep in range(1, args.epochs_stage1 + 1):
            tr = train_one_epoch(model_s1, tr16_s1_loader, opt_s1, device)
            va, gc = evaluate(model_s1, va16_s1_loader, device)
            hist1.append({"epoch": ep, "train_loss": tr, "val_loss": va, "val_gene_corr": gc})
            print(f"Stage1 Ep {ep:03d} | train={tr:.5f} | val={va:.5f} | corr={gc:.4f}")
            if va < best1:
                best1 = va
                torch.save({"model_state_dict": unwrap(model_s1).state_dict(), "genes": hvg_genes, "best_val": best1}, ckpt1)
    except KeyboardInterrupt:
        print(f"\n[WARN] Stage1 interrupted at ep {ep}. Continuing with best checkpoint so far.")

    stage1_df = pd.DataFrame(hist1)
    stage1_df.to_csv(os.path.join(out_dir, "stage1_history.csv"), index=False)

    # Transfer encoder weights stage1 -> stage2, excluding the gene decoder (different gene count).
    ck1 = safe_torch_load(ckpt1, map_location="cpu")
    ms2 = unwrap(model_s2).state_dict()
    for k in ms2:
        if k in ck1["model_state_dict"] and ck1["model_state_dict"][k].shape == ms2[k].shape and not k.startswith("gene_decoder"):
            ms2[k] = ck1["model_state_dict"][k]
    unwrap(model_s2).load_state_dict(ms2, strict=False)

    # Stage 2: fine-tune on the full gene set, plus any auxiliary resolutions requested.
    hist2, best2 = [], float("inf")
    ckpt2 = os.path.join(out_dir, "stage2_best.pt")
    ep = 0
    try:
        for ep in range(1, args.epochs_stage2 + 1):
            tr = train_one_epoch(model_s2, tr16_s2_loader, opt_s2, device, aux_loaders=aux_loaders, aux_weights=aux_weights)
            va, gc = evaluate(model_s2, va16_s2_loader, device)
            hist2.append({"epoch": ep, "train_loss": tr, "val_loss": va, "val_gene_corr": gc})
            print(f"Stage2 Ep {ep:03d} | train={tr:.5f} | val={va:.5f} | corr={gc:.4f}")
            if va < best2:
                best2 = va
                torch.save({"model_state_dict": unwrap(model_s2).state_dict(), "genes": common_genes, "best_val": best2}, ckpt2)
    except KeyboardInterrupt:
        print(f"\n[WARN] Stage2 interrupted at ep {ep}. Proceeding to inference with best checkpoint.")

    stage2_df = pd.DataFrame(hist2)
    stage2_df.to_csv(os.path.join(out_dir, "stage2_history.csv"), index=False)
    save_training_plots(stage1_df, stage2_df, out_dir)

    best = safe_torch_load(ckpt2, map_location="cpu")
    unwrap(model_s2).load_state_dict(best["model_state_dict"], strict=True)
    model_s2.eval()

    df_spot, df_gene, df_pred, _ = run_v2_inference(model_s2, v2_mem, v2_pos, v2_ad, common_genes, args.n_v2_infer, device)
    df_spot.to_csv(os.path.join(out_dir, "v2_spot_metrics.csv"), index=False)
    df_gene.to_csv(os.path.join(out_dir, "v2_gene_metrics.csv"), index=False)
    df_pred.to_csv(os.path.join(out_dir, "v2_predictions.csv"), index=False)

    gene_keep_list = sorted(df_pred["gene"].unique().tolist())
    bc_to_row = {bc: i for i, bc in enumerate(df_spot["barcode"].tolist())}
    gene_to_col = {g: i for i, g in enumerate(gene_keep_list)}
    all_pred_mat = np.zeros((len(df_spot), len(gene_keep_list)), dtype=np.float32)
    for _, r in df_pred.iterrows():
        ri, ci = bc_to_row.get(r["barcode"]), gene_to_col.get(r["gene"])
        if ri is not None and ci is not None:
            all_pred_mat[ri, ci] = r["pred_expr"]

    df_hd, hd_reg_info = run_hd_bin_comparison(hd_mem, v2_mem, hd16_pos, hd16_ad, df_spot, all_pred_mat, gene_keep_list, out_dir)
    if len(df_hd):
        df_hd.to_csv(os.path.join(out_dir, "hd_comparison.csv"), index=False)

    def mean_of(df, col):
        return float(df[col].mean()) if len(df) and col in df else np.nan

    def median_of(df, col):
        return float(df[col].median()) if len(df) and col in df else np.nan

    summary = {
        "pipeline": "baseline_direct",
        "n_common_genes": len(common_genes), "n_hvg": len(hvg_genes), "gpus_used": gpu_ids,
        "stage1_best_val_loss": float(stage1_df["val_loss"].min()) if len(stage1_df) else np.nan,
        "stage1_best_val_gene_corr": float(stage1_df["val_gene_corr"].max()) if len(stage1_df) else np.nan,
        "stage2_best_val_loss": float(stage2_df["val_loss"].min()) if len(stage2_df) else np.nan,
        "stage2_best_val_gene_corr": float(stage2_df["val_gene_corr"].max()) if len(stage2_df) else np.nan,
        "mean_bins_per_spot": mean_of(df_spot, "n_bins_used"),
        "spot_mean_pearson": mean_of(df_spot, "spot_log1p_pearson"),
        "spot_median_pearson": median_of(df_spot, "spot_log1p_pearson"),
        "spot_mean_spearman": mean_of(df_spot, "spot_log1p_spearman"),
        "spot_mean_rmse": mean_of(df_spot, "spot_log1p_rmse"),
        "gene_mean_pearson": mean_of(df_gene, "gene_pearson"),
        "gene_mean_spearman": mean_of(df_gene, "gene_spearman"),
        "gene_mean_rmse": mean_of(df_gene, "gene_rmse"),
        "hd_mean_pearson": mean_of(df_hd, "hd_pearson"),
        "hd_mean_spearman": mean_of(df_hd, "hd_spearman"),
        "hd_mean_rmse": mean_of(df_hd, "hd_rmse"),
        "hd_mean_bins_per_spot": mean_of(df_hd, "n_hd_bins_overlapping"),
        "hd_registration_inliers": hd_reg_info.get("n_inliers"),
        "hd_registration_n_matches": hd_reg_info.get("n_matches"),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description="Baseline: direct image-to-gene prediction (no signal bottleneck)")
    ap.add_argument("--config", type=str, default="configs/paths.yaml", help="Path to dataset paths YAML")
    ap.add_argument("--out_dir", type=str, default=None, help="Overrides output_dir from the config file")
    ap.add_argument("--gene_limit", type=int, default=DEFAULTS["gene_limit"])
    ap.add_argument("--hvg", type=int, default=DEFAULTS["hvg"])
    ap.add_argument("--epochs_stage1", type=int, default=DEFAULTS["epochs_stage1"])
    ap.add_argument("--epochs_stage2", type=int, default=DEFAULTS["epochs_stage2"])
    ap.add_argument("--batch_stage1", type=int, default=DEFAULTS["batch_stage1"])
    ap.add_argument("--batch_stage2", type=int, default=DEFAULTS["batch_stage2"])
    ap.add_argument("--lr_head", type=float, default=DEFAULTS["lr_head"])
    ap.add_argument("--lr_backbone", type=float, default=DEFAULTS["lr_backbone"])
    ap.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    ap.add_argument("--max_hd16", type=int, default=2000)
    ap.add_argument("--max_hd8", type=int, default=8000)
    ap.add_argument("--max_hd2", type=int, default=128000)
    ap.add_argument("--max_v2", type=int, default=100)
    ap.add_argument("--n_v2_infer", type=int, default=DEFAULTS["n_v2_infer"])
    ap.add_argument("--use_hd8", action="store_true")
    ap.add_argument("--use_hd2", action="store_true")
    ap.add_argument("--use_v2_train", action="store_true")
    ap.add_argument("--gpus", type=int, nargs="+", default=None, help="GPU IDs, e.g. --gpus 0 2. First is primary.")
    args = ap.parse_args()

    paths = load_dataset_paths(args.config)
    args.uni_weights = paths.uni_weights
    out_dir = args.out_dir or paths.output_dir

    if args.gpus:
        gpu_ids = args.gpus
    elif torch.cuda.is_available():
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = []
    device = torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids else torch.device("cpu")
    print(f"Primary device: {device}")
    if len(gpu_ids) > 1:
        print(f"DataParallel on: {gpu_ids}")

    seed_everything(SEED)
    os.makedirs(out_dir, exist_ok=True)

    shared = load_sample_bundle(paths, args.gene_limit)

    seed_everything(SEED)
    run(args, shared, out_dir, gpu_ids, device)
    print("Done.")


if __name__ == "__main__":
    main()
