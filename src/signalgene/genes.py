"""Gene selection utilities."""

import numpy as np
import pandas as pd
import scanpy as sc


def select_hvg(X: np.ndarray, gene_names, n_top: int = 3000):
    """Return (indices, names) of the top-n highly variable genes (Seurat v3 flavor)."""
    if n_top is None or n_top >= len(gene_names):
        return np.arange(len(gene_names)), list(gene_names)
    ad = sc.AnnData(X.copy())
    ad.var_names = pd.Index(gene_names)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=n_top, flavor="seurat_v3")
    idx = np.sort(np.where(ad.var["highly_variable"].to_numpy())[0])
    return idx, [gene_names[i] for i in idx]
