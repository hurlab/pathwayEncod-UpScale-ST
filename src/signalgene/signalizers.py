"""Signalizers: transforms from a gene-expression vector to a structured 1-D signal.

Used by the pathway-guided pipeline (scripts/train_pathway_guided.py) to build a short
signal target from biologically or statistically related gene groups, which the model
predicts and then decodes back into per-gene expression.
"""

import json
import os
from typing import Dict, Optional

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


class SignalizerBase:
    name = "base"

    def fit(self, X, gene_names):
        raise NotImplementedError

    def encode(self, X):
        raise NotImplementedError

    def decode(self, Z):
        raise NotImplementedError

    def state_dict(self):
        return {"name": self.name}


class CoexpressionOrderSignalizer(SignalizerBase):
    """Orders genes by hierarchical clustering on co-expression, then averages contiguous blocks."""

    name = "coexpression_order"

    def __init__(self, signal_len: int = 512):
        self.signal_len = signal_len

    def fit(self, X, gene_names):
        Xl = np.log1p(X)
        corr = np.nan_to_num(np.corrcoef(Xl.T), nan=0.0)
        dist = 1.0 - corr
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(np.clip(dist, 0.0, 2.0), checks=False)
        Z = linkage(condensed, method="average")
        self.order_ = leaves_list(Z).astype(np.int64)
        self.gene_names = list(gene_names)
        self.edges_ = np.linspace(0, len(gene_names), self.signal_len + 1).astype(int)
        self.signal_dim = self.signal_len
        return self

    def encode(self, X):
        Xl = np.log1p(X[:, self.order_])
        out = np.zeros((X.shape[0], self.signal_len), dtype=np.float32)
        for i in range(self.signal_len):
            a, b = self.edges_[i], self.edges_[i + 1]
            if b <= a:
                b = min(a + 1, Xl.shape[1])
            out[:, i] = Xl[:, a:b].mean(axis=1)
        return out

    def decode(self, Z):
        G = len(self.gene_names)
        Xord = np.zeros((Z.shape[0], G), dtype=np.float32)
        for i in range(self.signal_len):
            a, b = self.edges_[i], self.edges_[i + 1]
            if b <= a:
                b = min(a + 1, G)
            Xord[:, a:b] = Z[:, [i]]
        Xl = np.zeros_like(Xord)
        Xl[:, self.order_] = Xord
        return np.expm1(Xl).clip(min=0.0).astype(np.float32)

    def state_dict(self):
        return {
            "name": self.name,
            "signal_len": int(self.signal_len),
            "order": self.order_.tolist(),
            "gene_names": self.gene_names,
        }


class PathwayGroupSignalizer(SignalizerBase):
    """Averages expression within named pathway gene sets; falls back to contiguous slices
    when no pathway file is given or too few matching pathways are found."""

    name = "pathway_group"

    def __init__(self, pathway_json: Optional[str] = None, min_groups: int = 128):
        self.pathway_json = pathway_json
        self.min_groups = min_groups

    def fit(self, X, gene_names):
        self.gene_names = list(gene_names)
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        groups: Dict[str, np.ndarray] = {}
        if self.pathway_json is not None and os.path.exists(self.pathway_json):
            with open(self.pathway_json, "r") as f:
                pmap = json.load(f)
            for pname, glist in pmap.items():
                idx = [gene_to_idx[g] for g in glist if g in gene_to_idx]
                if idx:
                    groups[pname] = np.array(sorted(set(idx)), dtype=np.int64)
        if len(groups) < self.min_groups:
            edges = np.linspace(0, len(gene_names), self.min_groups + 1).astype(int)
            groups = {}
            for i in range(self.min_groups):
                a, b = edges[i], edges[i + 1]
                if b > a:
                    groups[f"slice_{i:03d}"] = np.arange(a, b, dtype=np.int64)
        self.group_names_ = list(groups.keys())
        self.groups_ = groups
        self.signal_dim = len(self.group_names_)
        return self

    def encode(self, X):
        Xl = np.log1p(X)
        out = np.zeros((X.shape[0], self.signal_dim), dtype=np.float32)
        for j, gname in enumerate(self.group_names_):
            out[:, j] = Xl[:, self.groups_[gname]].mean(axis=1)
        return out

    def decode(self, Z):
        G = len(self.gene_names)
        Xl = np.zeros((Z.shape[0], G), dtype=np.float32)
        counts = np.zeros((G,), dtype=np.float32)
        for j, gname in enumerate(self.group_names_):
            idx = self.groups_[gname]
            Xl[:, idx] += Z[:, [j]]
            counts[idx] += 1.0
        counts = np.clip(counts, 1.0, None)
        return np.expm1(Xl / counts[None, :]).clip(min=0.0).astype(np.float32)

    def state_dict(self):
        return {
            "name": self.name,
            "group_names": self.group_names_,
            "groups": {k: v.tolist() for k, v in self.groups_.items()},
            "gene_names": self.gene_names,
            "pathway_json": self.pathway_json,
        }


ALL_METHODS = ["coexpression_order", "pathway_group"]


def build_signalizer(method: str, signal_len: int, pathway_json: Optional[str]) -> SignalizerBase:
    if method == "coexpression_order":
        return CoexpressionOrderSignalizer(signal_len=signal_len)
    if method == "pathway_group":
        return PathwayGroupSignalizer(pathway_json=pathway_json, min_groups=min(128, signal_len))
    raise ValueError(f"Unknown signalizer method: {method}")
