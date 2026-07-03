"""HD <-> V2 image registration and the resulting bin-level comparison.

Registers the HD image onto the V2 image (ORB features + RANSAC affine), then compares
each predicted V2 spot against an area-weighted aggregate of the true HD bins that
overlap it.
"""

import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.spatial import cKDTree
from tqdm import tqdm

from .metrics import safe_pearson, safe_rmse, safe_spearman
from .patches import compute_bin_overlap_fraction, estimate_px_per_um

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def register_hd_to_v2(hd_mem: np.ndarray, v2_mem: np.ndarray, downsample: int = 16,
                       n_features: int = 10000, ransac_thresh: float = 3.0):
    """Register the HD H&E image onto the V2 H&E image with ORB features + RANSAC.

    Returns a (2,3) affine mapping HD full-res (col,row) -> V2 full-res (col,row), plus a
    dict of registration diagnostics (keypoint/match/inlier counts).
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("opencv-python is required for image registration")

    hd_s = hd_mem[::downsample, ::downsample]
    v2_s = v2_mem[::downsample, ::downsample]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hd_g = clahe.apply(cv2.cvtColor(hd_s, cv2.COLOR_RGB2GRAY))
    v2_g = clahe.apply(cv2.cvtColor(v2_s, cv2.COLOR_RGB2GRAY))

    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(hd_g, None)
    kp2, des2 = orb.detectAndCompute(v2_g, None)
    print(f"  ORB keypoints — HD: {len(kp1)}, V2: {len(kp2)}")
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        raise RuntimeError("Not enough ORB keypoints for registration")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
    print(f"  Matches (crossCheck): {len(matches)}")
    if len(matches) < 10:
        raise RuntimeError(f"Too few matches: {len(matches)}")

    pts_hd = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_v2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 4-DOF affine: rotation + uniform scale + translation, no shear.
    M_small, inliers = cv2.estimateAffinePartial2D(
        pts_hd, pts_v2, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh,
        maxIters=5000, confidence=0.99,
    )
    if M_small is None:
        raise RuntimeError("Affine estimation failed")

    n_inliers = int(inliers.sum()) if inliers is not None else 0
    print(f"  Inliers: {n_inliers}/{len(matches)}")

    M_full = M_small.copy()
    M_full[:, 2] *= downsample  # translation was estimated at the downsampled scale

    info = {
        "method": "orb_affine_partial2d",
        "n_keypoints_hd": len(kp1), "n_keypoints_v2": len(kp2),
        "n_matches": len(matches), "n_inliers": n_inliers,
        "downsample": downsample, "affine_2x3": M_full.tolist(),
    }
    return M_full.astype(np.float32), info


def invert_affine_2x3(M: np.ndarray) -> np.ndarray:
    A = np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float64)
    return np.linalg.inv(A)[:2, :].astype(np.float32)


def transform_rowcol(coords_rc: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a (2,3) affine to (N,2) (row,col) coordinates, returning (N,2) (row,col)."""
    pts = np.asarray(coords_rc, dtype=np.float32)
    xy1 = np.stack([pts[:, 1], pts[:, 0], np.ones(len(pts))], axis=0)
    out = M.astype(np.float32) @ xy1
    return np.stack([out[1], out[0]], axis=1).astype(np.float32)


def run_hd_bin_comparison(hd_mem: np.ndarray, v2_mem: np.ndarray, hd16_pos: pd.DataFrame, hd16_ad,
                           df_spot: pd.DataFrame, all_pred: np.ndarray, gene_keep: List[str],
                           method_out: str, tag: str = "") -> Tuple[pd.DataFrame, dict]:
    """For every predicted V2 spot, map it into HD pixel space, aggregate the true HD 16um
    bins overlapping it (area-weighted), and compare against the model's prediction."""
    if not CV2_AVAILABLE:
        print("  [HD comparison] Skipped — opencv not available.")
        return pd.DataFrame(), {}

    print("\n[HD comparison] Registering HD -> V2 ...")
    try:
        M_hd_to_v2, reg_info = register_hd_to_v2(hd_mem, v2_mem)
    except Exception as e:
        print(f"  [HD comparison] Registration failed: {e}")
        return pd.DataFrame(), {}

    suffix = f"_{tag}" if tag else ""
    with open(os.path.join(method_out, f"registration_info{suffix}.json"), "w") as f:
        json.dump(reg_info, f, indent=2)

    M_v2_to_hd = invert_affine_2x3(M_hd_to_v2)

    genes_in_hd = set(hd16_ad.var_names.astype(str).tolist())
    gene_keep_hd = [g for g in gene_keep if g in genes_in_hd]
    pred_idx = {g: i for i, g in enumerate(gene_keep)}
    print(f"  [HD comparison] Common genes (pred and HD): {len(gene_keep_hd)}")

    common_bcs = np.intersect1d(hd16_pos.index.astype(str), hd16_ad.obs_names.astype(str))
    hd16_pos_f = hd16_pos.loc[common_bcs]
    hd_coords = hd16_pos_f[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy(np.float32)

    X_hd = hd16_ad[common_bcs, :][:, gene_keep_hd].X
    X_hd = (X_hd.toarray() if issparse(X_hd) else np.asarray(X_hd)).astype(np.float32)

    tree_hd = cKDTree(hd_coords)
    hd_px_per_um = estimate_px_per_um(hd_coords, 16.0)
    bin_half_hd = 8.0 * hd_px_per_um

    # Estimate the V2 spot radius in HD pixel space by mapping cardinal points of a
    # representative spot circle through the inverse affine.
    v2_px_per_um = estimate_px_per_um(df_spot[["cy", "cx"]].to_numpy(np.float32), 55.0)
    v2_r_px = 27.5 * v2_px_per_um
    ref_cy, ref_cx = float(df_spot["cy"].median()), float(df_spot["cx"].median())
    circle_pts_v2 = np.array([
        [ref_cy, ref_cx + v2_r_px], [ref_cy, ref_cx - v2_r_px],
        [ref_cy + v2_r_px, ref_cx], [ref_cy - v2_r_px, ref_cx],
    ], dtype=np.float32)
    circle_pts_hd = transform_rowcol(circle_pts_v2, M_v2_to_hd)
    ref_hd = transform_rowcol(np.array([[ref_cy, ref_cx]], dtype=np.float32), M_v2_to_hd)
    spot_r_hd = float(np.mean(np.sqrt(np.sum((circle_pts_hd - ref_hd) ** 2, axis=1))))
    search_r_hd = spot_r_hd + bin_half_hd * 1.415

    print(f"  [HD comparison] V2 spot r={v2_r_px:.1f}px -> HD r={spot_r_hd:.1f}px  bin_half={bin_half_hd:.1f}px")

    rows = []
    for spot_idx, row in tqdm(df_spot.iterrows(), total=len(df_spot), desc="HD comparison", leave=False):
        v2_cy, v2_cx = float(row["cy"]), float(row["cx"])
        hd_rc = transform_rowcol(np.array([[v2_cy, v2_cx]]), M_v2_to_hd)
        hd_cy, hd_cx = float(hd_rc[0, 0]), float(hd_rc[0, 1])

        nearby = tree_hd.query_ball_point([hd_cy, hd_cx], r=search_r_hd)
        if not nearby:
            continue

        true_wsum = np.zeros(len(gene_keep_hd), dtype=np.float64)
        weight_sum, n_bins = 0.0, 0
        for hi in nearby:
            frac = compute_bin_overlap_fraction(hd_coords[hi, 0], hd_coords[hi, 1], bin_half_hd, hd_cy, hd_cx, spot_r_hd)
            if frac < 0.01:
                continue
            true_wsum += frac * X_hd[hi]
            weight_sum += frac
            n_bins += 1
        if weight_sum < 0.01:
            continue

        true_agg = (true_wsum / weight_sum).astype(np.float32)
        pred_agg = np.array([all_pred[spot_idx, pred_idx[g]] for g in gene_keep_hd], dtype=np.float32)
        lt, lp = np.log1p(true_agg), np.log1p(pred_agg)
        rows.append({
            "barcode": row["barcode"],
            "v2_cy": v2_cy, "v2_cx": v2_cx, "hd_cy": hd_cy, "hd_cx": hd_cx,
            "n_hd_bins_overlapping": n_bins, "total_overlap_weight": float(weight_sum),
            "hd_pearson": safe_pearson(lt, lp), "hd_spearman": safe_spearman(lt, lp), "hd_rmse": safe_rmse(lt, lp),
        })

    df_hd = pd.DataFrame(rows)
    if len(df_hd):
        print(f"  [HD comparison] Spots compared: {len(df_hd)} | "
              f"mean Pearson={df_hd['hd_pearson'].mean():.4f} | "
              f"mean Spearman={df_hd['hd_spearman'].mean():.4f} | "
              f"mean RMSE={df_hd['hd_rmse'].mean():.4f}")
    return df_hd, reg_info
