"""Patch dataset shared by both pipelines.

When a signalizer is provided, each item also carries its encoded signal target
(pathway-guided pipeline). Without one, items only carry raw expression (baseline).
"""

import numpy as np
import torch
from scipy.sparse import issparse
from torch.utils.data import Dataset

from .patches import build_views, estimate_px_per_um


class PatchDataset(Dataset):
    def __init__(self, mem_rgb, coords_df, adata, gene_names, scale_um, center_box,
                 signalizer=None, max_samples=None, seed=42):
        common = np.intersect1d(
            coords_df.index.astype(str).to_numpy(), adata.obs_names.astype(str).to_numpy()
        )
        coords_df = coords_df.loc[common]
        adata = adata[common].copy()
        X = adata[:, gene_names].X
        X = X.toarray() if issparse(X) else np.asarray(X)
        keep = X.sum(axis=1) > 0
        coords_df = coords_df.iloc[np.where(keep)[0]]
        X = X[keep]

        if max_samples and len(X) > max_samples:
            idx = np.random.default_rng(seed).choice(len(X), size=max_samples, replace=False)
            coords_df = coords_df.iloc[idx]
            X = X[idx]

        self.mem_rgb = mem_rgb
        self.coords = coords_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy(np.float32)
        self.barcodes = coords_df.index.astype(str).to_numpy()
        self.X = X.astype(np.float32)
        self.scale_um = int(scale_um)
        self.center_box = int(center_box)
        self.px_per_um = estimate_px_per_um(self.coords, scale_um)

        self.signalizer = signalizer
        self.Z = signalizer.encode(self.X).astype(np.float32) if signalizer is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        cy, cx = self.coords[idx]
        fine, mid, coarse, masked = build_views(
            self.mem_rgb, cy, cx, self.scale_um, self.center_box, self.px_per_um
        )
        item = {
            "fine": fine, "mid": mid, "coarse": coarse, "masked": masked,
            "expr": torch.tensor(self.X[idx], dtype=torch.float32),
            "barcode": self.barcodes[idx],
        }
        if self.Z is not None:
            item["signal"] = torch.tensor(self.Z[idx], dtype=torch.float32)
        return item
