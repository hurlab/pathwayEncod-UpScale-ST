"""Loads all images, expression matrices, and coordinates for one paired HD/V2 sample
into a single dict that both pipelines' training scripts consume."""

from typing import List

import numpy as np

from .config import DatasetPaths
from .io import load_expression_flexible, load_hd_coords, load_v2_coords, open_memmap_rgb


def load_sample_bundle(paths: DatasetPaths, gene_limit: int) -> dict:
    resolved = paths.resolve_required_files()

    print("Loading whole-slide images...")
    _, hd_mem = open_memmap_rgb(resolved["hd_image"])
    _, v2_mem = open_memmap_rgb(resolved["v2_image"])

    print("Loading expression matrices and coordinates...")
    hd2_ad = load_expression_flexible(resolved["hd_h5_002um"])
    hd8_ad = load_expression_flexible(resolved["hd_h5_008um"])
    hd16_ad = load_expression_flexible(resolved["hd_h5_016um"])
    v2_ad = load_expression_flexible(resolved["v2_expression"])

    hd2_pos = load_hd_coords(resolved["hd_coords_002um"])
    hd8_pos = load_hd_coords(resolved["hd_coords_008um"])
    hd16_pos = load_hd_coords(resolved["hd_coords_016um"])
    v2_pos = load_v2_coords(resolved["v2_positions"])

    common_genes: List[str] = np.intersect1d(hd16_ad.var_names.astype(str), v2_ad.var_names.astype(str))
    common_genes = np.intersect1d(common_genes, hd8_ad.var_names.astype(str))
    common_genes = np.intersect1d(common_genes, hd2_ad.var_names.astype(str)).tolist()
    if gene_limit and len(common_genes) > gene_limit:
        common_genes = common_genes[:gene_limit]
    if len(common_genes) < 100:
        raise RuntimeError("Too few common genes across HD and V2 samples.")
    print(f"Common genes: {len(common_genes)}")

    return {
        "hd_mem": hd_mem, "v2_mem": v2_mem,
        "hd2_ad": hd2_ad, "hd8_ad": hd8_ad, "hd16_ad": hd16_ad, "v2_ad": v2_ad,
        "hd2_pos": hd2_pos, "hd8_pos": hd8_pos, "hd16_pos": hd16_pos, "v2_pos": v2_pos,
        "common_genes": common_genes,
    }
