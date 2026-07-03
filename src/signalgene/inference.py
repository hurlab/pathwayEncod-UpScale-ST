"""Weighted area-overlap inference: predicts gene expression for a V2 spot by tiling it
with model-native bins, weighting each bin's prediction by its fractional area overlap
with the (circular) V2 spot footprint.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from tqdm import tqdm

from .constants import HD_CENTER_BOX, SEED
from .metrics import safe_pearson, safe_rmse, safe_spearman
from .patches import build_views, compute_bin_overlap_fraction, estimate_px_per_um


@torch.no_grad()
def run_v2_inference(model, v2_mem, v2_pos, v2_ad, model_genes, n_spots, device
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Returns (df_spot, df_gene, df_pred, df_signal). df_signal is None for models
    that don't produce an intermediate signal."""
    model.eval()
    common = np.intersect1d(v2_pos.index.astype(str).to_numpy(), v2_ad.obs_names.astype(str).to_numpy())
    rng = np.random.default_rng(SEED)
    sel = rng.choice(common, size=min(n_spots, len(common)), replace=False)
    coords = v2_pos.loc[sel][["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy(np.float32)
    px_per_um = estimate_px_per_um(coords, 55)

    spot_r_px = 27.5 * px_per_um
    bin_size_px = 16.0 * px_per_um
    bin_half = bin_size_px / 2.0
    step = bin_size_px
    n_steps = int(np.ceil((spot_r_px + bin_half) / step))

    genes_in_v2 = set(v2_ad.var_names.astype(str).tolist())
    gene_keep = [g for g in model_genes if g in genes_in_v2]
    gene_to_model_idx = {g: i for i, g in enumerate(model_genes)}
    n_genes = len(gene_keep)

    all_true = np.zeros((len(sel), n_genes), dtype=np.float32)
    all_pred = np.zeros((len(sel), n_genes), dtype=np.float32)
    rows_spot, rows_signal = [], []
    has_signal = None

    for spot_idx, bc in enumerate(tqdm(sel, desc="v2 weighted inference", leave=False)):
        spot_cy = float(v2_pos.loc[bc, "pxl_row_in_fullres"])
        spot_cx = float(v2_pos.loc[bc, "pxl_col_in_fullres"])

        pred_sum = np.zeros(len(model_genes), dtype=np.float64)
        signal_sum = None
        weight_sum = 0.0
        n_bins_used = 0
        offsets = np.arange(-n_steps, n_steps + 1) * step

        def forward_bin(cy, cx):
            fine, mid, coarse, masked = build_views(v2_mem, cy, cx, 16, HD_CENTER_BOX[16], px_per_um)
            out = model(
                fine.unsqueeze(0).to(device), mid.unsqueeze(0).to(device),
                coarse.unsqueeze(0).to(device), masked.unsqueeze(0).to(device),
            )
            if isinstance(out, tuple):
                z_hat, x_hat = out
                return x_hat.squeeze(0).cpu().numpy().astype(np.float32), z_hat.squeeze(0).cpu().numpy().astype(np.float32)
            return out.squeeze(0).cpu().numpy().astype(np.float32), None

        for dy in offsets:
            for dx in offsets:
                if (dy ** 2 + dx ** 2) > (spot_r_px + bin_half * 1.415) ** 2:
                    continue
                bin_cy, bin_cx = spot_cy + dy, spot_cx + dx
                frac = compute_bin_overlap_fraction(bin_cy, bin_cx, bin_half, spot_cy, spot_cx, spot_r_px)
                if frac < 0.01:
                    continue
                try:
                    x_np, z_np = forward_bin(bin_cy, bin_cx)
                except Exception:
                    continue
                pred_sum += frac * x_np
                if z_np is not None:
                    signal_sum = frac * z_np if signal_sum is None else signal_sum + frac * z_np
                weight_sum += frac
                n_bins_used += 1

        if weight_sum < 0.01:
            x_np, z_np = forward_bin(spot_cy, spot_cx)
            pred_sum = x_np.astype(np.float64)
            signal_sum = z_np
            weight_sum = 1.0
            n_bins_used = 1

        if has_signal is None:
            has_signal = signal_sum is not None

        pred_final = pred_sum.astype(np.float32)
        signal_final = (signal_sum / weight_sum).astype(np.float32) if signal_sum is not None else None

        true_x = v2_ad[bc, gene_keep].X
        true_x = true_x.toarray().ravel() if issparse(true_x) else np.asarray(true_x).ravel()
        pred_x = np.array([pred_final[gene_to_model_idx[g]] for g in gene_keep], dtype=np.float32)

        all_true[spot_idx] = true_x
        all_pred[spot_idx] = pred_x

        lt, lp = np.log1p(true_x), np.log1p(pred_x)
        row = {
            "barcode": bc, "cy": spot_cy, "cx": spot_cx,
            "n_bins_used": n_bins_used, "total_weight": float(weight_sum),
            "pred_expr_sum": float(np.sum(pred_x)), "true_expr_sum": float(np.sum(true_x)),
            "spot_log1p_pearson": safe_pearson(lt, lp),
            "spot_log1p_spearman": safe_spearman(lt, lp),
            "spot_log1p_rmse": safe_rmse(lt, lp),
        }
        if signal_final is not None:
            row["pred_signal_mean"] = float(np.mean(signal_final))
            row["pred_signal_std"] = float(np.std(signal_final))
            for i, sv in enumerate(signal_final):
                rows_signal.append({"barcode": bc, "signal_index": i, "pred_signal": float(sv)})
        rows_spot.append(row)

    rows_gene = []
    for gi, g in enumerate(tqdm(gene_keep, desc="gene metrics", leave=False)):
        gt, gp = np.log1p(all_true[:, gi]), np.log1p(all_pred[:, gi])
        rows_gene.append({
            "gene": g,
            "gene_pearson": safe_pearson(gt, gp),
            "gene_spearman": safe_spearman(gt, gp),
            "gene_rmse": safe_rmse(gt, gp),
            "true_mean": float(np.mean(all_true[:, gi])),
            "pred_mean": float(np.mean(all_pred[:, gi])),
        })

    rows_pred = [
        {"barcode": sel[si], "gene": gene_keep[gi],
         "true_expr": float(all_true[si, gi]), "pred_expr": float(all_pred[si, gi])}
        for si in range(len(sel)) for gi in range(n_genes)
    ]

    df_signal = pd.DataFrame(rows_signal) if has_signal else None
    return pd.DataFrame(rows_spot), pd.DataFrame(rows_gene), pd.DataFrame(rows_pred), df_signal
