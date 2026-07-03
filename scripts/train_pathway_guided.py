#!/usr/bin/env python3
"""Pathway-guided pipeline: image -> structured signal -> gene expression.

The model predicts a structured intermediate signal built by a signalizer
(see signalgene/signalizers.py):
  - coexpression_order: genes ordered by co-expression clustering, averaged into blocks
  - pathway_group:      genes averaged within named pathway sets (or contiguous
                         fallback slices if no pathway file is supplied)
The signal is then decoded back into per-gene predictions.
See scripts/train_baseline_direct.py for the no-signal baseline.

Example:
    python scripts/train_pathway_guided.py --method coexpression_order --use_hd8 --use_hd2 --gpus 0 2 3
    python scripts/train_pathway_guided.py --method pathway_group --pathway_json pathways.json --use_hd8 --use_hd2
    python scripts/train_pathway_guided.py --run_all_methods --pathway_json pathways.json --use_hd8 --use_hd2
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from signalgene.bundle import load_sample_bundle
from signalgene.config import load_dataset_paths
from signalgene.constants import DEFAULTS, HD_CENTER_BOX, SEED, V2_CENTER_BOX
from signalgene.datasets import PatchDataset
from signalgene.engine import evaluate, train_one_epoch
from signalgene.genes import select_hvg
from signalgene.inference import run_v2_inference
from signalgene.models import ImageToSignalGeneModel
from signalgene.plotting import save_training_plots
from signalgene.registration import run_hd_bin_comparison
from signalgene.signalizers import ALL_METHODS, build_signalizer
from signalgene.utils import safe_torch_load, seed_everything, unwrap, wrap_model


def build_optimizer(model, lr_head, lr_backbone, weight_decay):
    base = unwrap(model)
    backbone_params = [p for n, p in base.named_parameters() if p.requires_grad and n.startswith("encoder.backbone")]
    head_params = [p for n, p in base.named_parameters() if p.requires_grad and not n.startswith("encoder.backbone")]
    return torch.optim.AdamW(
        [{"params": head_params, "lr": lr_head}, {"params": backbone_params, "lr": lr_backbone}],
        weight_decay=weight_decay,
    )


def run_method(method, args, shared, root_out_dir, gpu_ids, device):
    method_out = os.path.join(root_out_dir, method)
    os.makedirs(method_out, exist_ok=True)

    hd_mem, v2_mem = shared["hd_mem"], shared["v2_mem"]
    hd2_ad, hd8_ad, hd16_ad, v2_ad = shared["hd2_ad"], shared["hd8_ad"], shared["hd16_ad"], shared["v2_ad"]
    hd2_pos, hd8_pos, hd16_pos, v2_pos = shared["hd2_pos"], shared["hd8_pos"], shared["hd16_pos"], shared["v2_pos"]
    common_genes = shared["common_genes"]

    print(f"\n================ METHOD: {method} ================")
    X16_all = hd16_ad[:, common_genes].X
    X16_all = X16_all.toarray() if issparse(X16_all) else np.asarray(X16_all)
    idx_hvg, hvg_genes = select_hvg(X16_all, common_genes, n_top=min(args.hvg, len(common_genes)))

    sig_s1 = build_signalizer(method, args.signal_len, args.pathway_json).fit(X16_all[:, idx_hvg], hvg_genes)
    sig_s2 = build_signalizer(method, args.signal_len, args.pathway_json).fit(X16_all, common_genes)
    for tag, sig in [("stage1", sig_s1), ("stage2", sig_s2)]:
        with open(os.path.join(method_out, f"signalizer_{tag}_{method}.json"), "w") as f:
            json.dump(sig.state_dict(), f)

    ds16_s1 = PatchDataset(hd_mem, hd16_pos, hd16_ad, hvg_genes, 16, HD_CENTER_BOX[16], signalizer=sig_s1, max_samples=args.max_hd16)
    ds16_s2 = PatchDataset(hd_mem, hd16_pos, hd16_ad, common_genes, 16, HD_CENTER_BOX[16], signalizer=sig_s2, max_samples=args.max_hd16)
    ds8_s2 = PatchDataset(hd_mem, hd8_pos, hd8_ad, common_genes, 8, HD_CENTER_BOX[8], signalizer=sig_s2, max_samples=args.max_hd8) if args.use_hd8 else None
    ds2_s2 = PatchDataset(hd_mem, hd2_pos, hd2_ad, common_genes, 2, HD_CENTER_BOX[2], signalizer=sig_s2, max_samples=args.max_hd2) if args.use_hd2 else None
    dsv2_s2 = PatchDataset(v2_mem, v2_pos, v2_ad, common_genes, 55, V2_CENTER_BOX, signalizer=sig_s2, max_samples=args.max_v2) if args.use_v2_train else None

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

    model_s1 = wrap_model(ImageToSignalGeneModel(args.uni_weights, ds16_s1.Z.shape[1], len(hvg_genes)), gpu_ids, device)
    model_s2 = wrap_model(ImageToSignalGeneModel(args.uni_weights, ds16_s2.Z.shape[1], len(common_genes)), gpu_ids, device)
    if len(gpu_ids) > 1:
        print(f"  Using DataParallel on GPUs: {gpu_ids}")

    opt_s1 = build_optimizer(model_s1, args.lr_head, args.lr_backbone, args.weight_decay)
    opt_s2 = build_optimizer(model_s2, args.lr_head, args.lr_backbone, args.weight_decay)

    # Stage 1: pretrain on HVGs only.
    hist1, best1 = [], float("inf")
    ckpt1 = os.path.join(method_out, f"stage1_best_{method}.pt")
    ep = 0
    try:
        for ep in range(1, args.epochs_stage1 + 1):
            tr = train_one_epoch(model_s1, tr16_s1_loader, opt_s1, device)
            va, gc = evaluate(model_s1, va16_s1_loader, device)
            hist1.append({"epoch": ep, "train_loss": tr, "val_loss": va, "val_gene_corr": gc})
            print(f"Stage1 {method} Ep {ep:03d} | train={tr:.5f} | val={va:.5f} | corr={gc:.4f}")
            if va < best1:
                best1 = va
                torch.save({
                    "model_state_dict": unwrap(model_s1).state_dict(), "method": method, "genes": hvg_genes,
                    "signal_dim": int(ds16_s1.Z.shape[1]), "signalizer_state": sig_s1.state_dict(), "best_val": best1,
                }, ckpt1)
    except KeyboardInterrupt:
        print(f"\n[WARN] Stage1 interrupted at ep {ep}. Continuing to Stage2 with best checkpoint.")

    stage1_df = pd.DataFrame(hist1)
    stage1_df.to_csv(os.path.join(method_out, f"stage1_history_{method}.csv"), index=False)

    # Transfer encoder weights stage1 -> stage2 (skip the gene decoder and the signalizer's
    # final projection, since gene count / signal composition differ between stages).
    ck1 = safe_torch_load(ckpt1, map_location="cpu")
    ms1 = ck1["model_state_dict"]
    ms2 = unwrap(model_s2).state_dict()
    for k in ms2:
        if k in ms1 and ms1[k].shape == ms2[k].shape and not k.startswith("gene_decoder") and not k.startswith("signal_head.fc.2"):
            ms2[k] = ms1[k]
    unwrap(model_s2).load_state_dict(ms2, strict=False)

    # Stage 2: fine-tune on the full gene set, plus any auxiliary resolutions requested.
    hist2, best2 = [], float("inf")
    ckpt2 = os.path.join(method_out, f"stage2_best_{method}.pt")
    ep = 0
    try:
        for ep in range(1, args.epochs_stage2 + 1):
            tr = train_one_epoch(model_s2, tr16_s2_loader, opt_s2, device, aux_loaders=aux_loaders, aux_weights=aux_weights)
            va, gc = evaluate(model_s2, va16_s2_loader, device)
            hist2.append({"epoch": ep, "train_loss": tr, "val_loss": va, "val_gene_corr": gc})
            print(f"Stage2 {method} Ep {ep:03d} | train={tr:.5f} | val={va:.5f} | corr={gc:.4f}")
            if va < best2:
                best2 = va
                torch.save({
                    "model_state_dict": unwrap(model_s2).state_dict(), "method": method, "genes": common_genes,
                    "signal_dim": int(ds16_s2.Z.shape[1]), "signalizer_state": sig_s2.state_dict(), "best_val": best2,
                }, ckpt2)
    except KeyboardInterrupt:
        print(f"\n[WARN] Stage2 interrupted at ep {ep}. Proceeding to inference with best checkpoint.")

    stage2_df = pd.DataFrame(hist2)
    stage2_df.to_csv(os.path.join(method_out, f"stage2_history_{method}.csv"), index=False)
    save_training_plots(stage1_df, stage2_df, method_out, method)

    best = safe_torch_load(ckpt2, map_location="cpu")
    unwrap(model_s2).load_state_dict(best["model_state_dict"], strict=True)
    model_s2.eval()

    df_spot, df_gene, df_pred, df_signal = run_v2_inference(model_s2, v2_mem, v2_pos, v2_ad, common_genes, args.n_v2_infer, device)
    df_spot.to_csv(os.path.join(method_out, f"v2_spot_metrics_{method}.csv"), index=False)
    df_gene.to_csv(os.path.join(method_out, f"v2_gene_metrics_{method}.csv"), index=False)
    df_pred.to_csv(os.path.join(method_out, f"v2_predictions_{method}.csv"), index=False)
    if df_signal is not None:
        df_signal.to_csv(os.path.join(method_out, f"v2_signal_{method}.csv"), index=False)

    gene_keep_list = sorted(df_pred["gene"].unique().tolist())
    bc_to_row = {bc: i for i, bc in enumerate(df_spot["barcode"].tolist())}
    gene_to_col = {g: i for i, g in enumerate(gene_keep_list)}
    all_pred_mat = np.zeros((len(df_spot), len(gene_keep_list)), dtype=np.float32)
    for _, r in df_pred.iterrows():
        ri, ci = bc_to_row.get(r["barcode"]), gene_to_col.get(r["gene"])
        if ri is not None and ci is not None:
            all_pred_mat[ri, ci] = r["pred_expr"]

    print("\n[HD comparison] Starting HD-registered evaluation ...")
    df_hd, hd_reg_info = run_hd_bin_comparison(hd_mem, v2_mem, hd16_pos, hd16_ad, df_spot, all_pred_mat, gene_keep_list, method_out, tag=method)
    if len(df_hd):
        df_hd.to_csv(os.path.join(method_out, f"hd_comparison_{method}.csv"), index=False)

    def mean_of(df, col):
        return float(df[col].mean()) if len(df) and col in df else np.nan

    def median_of(df, col):
        return float(df[col].median()) if len(df) and col in df else np.nan

    summary = {
        "method": method,
        "signal_len_requested": int(args.signal_len),
        "stage1_signal_dim": int(ds16_s1.Z.shape[1]),
        "stage2_signal_dim": int(ds16_s2.Z.shape[1]),
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
    with open(os.path.join(method_out, f"summary_{method}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Pathway-guided pipeline: image -> signal -> gene expression")
    ap.add_argument("--method", type=str, default="coexpression_order", choices=ALL_METHODS)
    ap.add_argument("--run_all_methods", action="store_true")
    ap.add_argument("--config", type=str, default="configs/paths.yaml", help="Path to dataset paths YAML")
    ap.add_argument("--out_dir", type=str, default=None, help="Overrides output_dir from the config file")
    ap.add_argument("--gene_limit", type=int, default=DEFAULTS["gene_limit"])
    ap.add_argument("--hvg", type=int, default=DEFAULTS["hvg"])
    ap.add_argument("--signal_len", type=int, default=DEFAULTS["signal_len"])
    ap.add_argument("--epochs_stage1", type=int, default=DEFAULTS["epochs_stage1"])
    ap.add_argument("--epochs_stage2", type=int, default=DEFAULTS["epochs_stage2"])
    ap.add_argument("--batch_stage1", type=int, default=DEFAULTS["batch_stage1"])
    ap.add_argument("--batch_stage2", type=int, default=DEFAULTS["batch_stage2"])
    ap.add_argument("--lr_head", type=float, default=DEFAULTS["lr_head"])
    ap.add_argument("--lr_backbone", type=float, default=DEFAULTS["lr_backbone"])
    ap.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    ap.add_argument("--max_hd16", type=int, default=1000)
    ap.add_argument("--max_hd8", type=int, default=4000)
    ap.add_argument("--max_hd2", type=int, default=16000)
    ap.add_argument("--max_v2", type=int, default=100)
    ap.add_argument("--n_v2_infer", type=int, default=DEFAULTS["n_v2_infer"])
    ap.add_argument("--use_hd8", action="store_true")
    ap.add_argument("--use_hd2", action="store_true")
    ap.add_argument("--use_v2_train", action="store_true")
    ap.add_argument("--pathway_json", type=str, default=None, help="JSON file mapping pathway name -> gene list")
    ap.add_argument("--gpus", type=int, nargs="+", default=None, help="GPU IDs to use, e.g. --gpus 0 2 3. First is primary. Omit to use all available or CPU.")
    args = ap.parse_args()

    paths = load_dataset_paths(args.config)
    args.uni_weights = paths.uni_weights
    out_dir = args.out_dir or paths.output_dir

    if args.gpus is not None:
        gpu_ids = args.gpus
    elif torch.cuda.is_available():
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = []

    if gpu_ids:
        device = torch.device(f"cuda:{gpu_ids[0]}")
        print(f"Primary device : cuda:{gpu_ids[0]}")
        if len(gpu_ids) > 1:
            print(f"DataParallel on: {gpu_ids}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    seed_everything(SEED)
    os.makedirs(out_dir, exist_ok=True)

    shared = load_sample_bundle(paths, args.gene_limit)

    methods = ALL_METHODS if args.run_all_methods else [args.method]
    results = []
    for method in methods:
        seed_everything(SEED)
        summary = run_method(method, args, shared, out_dir, gpu_ids, device)
        results.append(summary)
        print(json.dumps(summary, indent=2))

    pd.DataFrame(results).to_csv(os.path.join(out_dir, "all_methods_comparison.csv"), index=False)
    print("Done.")


if __name__ == "__main__":
    main()
