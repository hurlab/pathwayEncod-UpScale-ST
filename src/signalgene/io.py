"""Loading whole-slide images, expression matrices, and spot/bin coordinates."""

import numpy as np
import pandas as pd
import scanpy as sc
import tifffile


def load_expression_flexible(path: str):
    """Load a 10x expression matrix from either an .h5 file or an mtx directory."""
    if path.endswith(".h5"):
        ad = sc.read_10x_h5(path)
    else:
        try:
            ad = sc.read_10x_mtx(path, var_names="gene_symbols", make_unique=True)
        except Exception:
            ad = sc.read_10x_mtx(path, var_names="gene_ids", make_unique=True)
    ad.obs_names = ad.obs_names.astype(str)
    ad.var_names_make_unique()
    return ad


def load_hd_coords(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(
            path,
            header=None,
            names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"],
        )
    df["barcode"] = df["barcode"].astype(str)
    df = df.set_index("barcode")
    if "in_tissue" in df.columns:
        df["in_tissue"] = pd.to_numeric(df["in_tissue"], errors="coerce").fillna(0).astype(int)
        df = df[df["in_tissue"] == 1]
    df["pxl_row_in_fullres"] = pd.to_numeric(df["pxl_row_in_fullres"], errors="coerce")
    df["pxl_col_in_fullres"] = pd.to_numeric(df["pxl_col_in_fullres"], errors="coerce")
    return df.dropna(subset=["pxl_row_in_fullres", "pxl_col_in_fullres"])


def load_v2_coords(path: str) -> pd.DataFrame:
    cols = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    df = pd.read_csv(path)
    if not set(cols).issubset(df.columns):
        df = pd.read_csv(path, header=None, names=cols)
    df["barcode"] = df["barcode"].astype(str)
    df = df.set_index("barcode")
    df["in_tissue"] = pd.to_numeric(df["in_tissue"], errors="coerce").fillna(0).astype(int)
    df = df[df["in_tissue"] == 1]
    df["pxl_row_in_fullres"] = pd.to_numeric(df["pxl_row_in_fullres"], errors="coerce")
    df["pxl_col_in_fullres"] = pd.to_numeric(df["pxl_col_in_fullres"], errors="coerce")
    return df.dropna(subset=["pxl_row_in_fullres", "pxl_col_in_fullres"])


def open_memmap_rgb(path: str):
    """Open a whole-slide TIFF/BTF as a memory-mapped uint8 RGB array."""
    tif = tifffile.TiffFile(path)
    mem = tif.pages[0].asarray(out="memmap")
    if mem.ndim == 2:
        mem = np.stack([mem, mem, mem], axis=-1)
    if mem.shape[-1] > 3:
        mem = mem[..., :3]
    if mem.dtype != np.uint8:
        vmax = float(np.max(mem)) if np.max(mem) > 0 else 1.0
        mem = np.clip((mem / vmax) * 255.0, 0, 255).astype(np.uint8)
    return tif, mem
