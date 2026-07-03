"""Multi-scale patch extraction around a bin/spot center in whole-slide pixel space."""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from sklearn.neighbors import NearestNeighbors

from .constants import COARSE_MULT, FINE_MULT, MAX_SRC, MID_MULT, MIN_SRC, NORMALIZE, PATCH_SIZE


def robust_extract_patch(mem_rgb: np.ndarray, cy: float, cx: float, patch_size: int = 224) -> np.ndarray:
    """Extract a patch_size x patch_size crop centered at (cy, cx), zero-padding at slide edges."""
    h, w = mem_rgb.shape[:2]
    half = patch_size // 2
    y0, y1 = int(round(cy)) - half, int(round(cy)) + half
    x0, x1 = int(round(cx)) - half, int(round(cx)) + half
    sy0, sy1 = max(0, y0), min(h, y1)
    sx0, sx1 = max(0, x0), min(w, x1)
    out = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    if sy1 <= sy0 or sx1 <= sx0:
        return out
    src = mem_rgb[sy0:sy1, sx0:sx1, :]
    dy0, dx0 = sy0 - y0, sx0 - x0
    dy1 = min(patch_size, dy0 + src.shape[0])
    dx1 = min(patch_size, dx0 + src.shape[1])
    out[dy0:dy1, dx0:dx1, :] = src[: (dy1 - dy0), : (dx1 - dx0), :]
    return out


def extract_resized(mem_rgb: np.ndarray, cy: float, cx: float, src_size, out_size: int = 224) -> torch.Tensor:
    raw = robust_extract_patch(mem_rgb, cy, cx, int(src_size))
    return TF.resize(TF.to_tensor(raw).float(), [out_size, out_size], antialias=True)


def estimate_px_per_um(coords_xy: np.ndarray, scale_um: float) -> float:
    """Estimate pixels-per-micron from the median nearest-neighbor spacing between spot/bin centers."""
    if len(coords_xy) < 10:
        return 1.0
    sample = coords_xy[: min(5000, len(coords_xy))]
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(sample)
    dist, _ = nn.kneighbors(sample)
    median_spacing = float(np.median(dist[:, 1]))
    return max(median_spacing / float(scale_um), 0.25)


def build_views(mem_rgb, cy, cx, scale_um, center_box, px_per_um):
    """Build the fine/mid/coarse context views plus the exact-bin-size center crop."""
    target_px = max(int(round(scale_um * px_per_um)), center_box)
    src_fine = int(np.clip(round(target_px * FINE_MULT), MIN_SRC, MAX_SRC))
    src_mid = int(np.clip(round(target_px * MID_MULT), MIN_SRC, MAX_SRC))
    src_coarse = int(np.clip(round(target_px * COARSE_MULT), MIN_SRC, MAX_SRC))

    fine_t = extract_resized(mem_rgb, cy, cx, src_fine, PATCH_SIZE)
    mid_t = extract_resized(mem_rgb, cy, cx, src_mid, PATCH_SIZE)
    coarse_t = extract_resized(mem_rgb, cy, cx, src_coarse, PATCH_SIZE)
    # Center view uses the true bin footprint (center_box px), not a scaled context window.
    masked_t = extract_resized(mem_rgb, cy, cx, center_box, PATCH_SIZE)
    return NORMALIZE(fine_t), NORMALIZE(mid_t), NORMALIZE(coarse_t), NORMALIZE(masked_t)


def compute_bin_overlap_fraction(bin_cy, bin_cx, bin_half, spot_cy, spot_cx, spot_r, n: int = 20) -> float:
    """Fraction of a square bin's area that falls inside a circular spot, via grid sampling."""
    ys = np.linspace(bin_cy - bin_half, bin_cy + bin_half, n)
    xs = np.linspace(bin_cx - bin_half, bin_cx + bin_half, n)
    yy, xx = np.meshgrid(ys, xs)
    inside = ((yy - spot_cy) ** 2 + (xx - spot_cx) ** 2 <= spot_r ** 2).sum()
    return float(inside) / (n * n)
