"""Shared constants for patch extraction and default training hyperparameters."""

from torchvision import transforms

SEED = 42
PATCH_SIZE = 224
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]
NORMALIZE = transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD)

# Bin center-crop size (in pixels) per bin resolution, plus the V2 spot equivalent.
HD_CENTER_BOX = {2: 7, 8: 29, 16: 59}
V2_CENTER_BOX = 161

# Multi-scale context view sizes, expressed as multiples of the bin's native pixel size.
FINE_MULT = 1.0
MID_MULT = 2.5
COARSE_MULT = 5.0
MIN_SRC = 32
MAX_SRC = 896

DEFAULTS = dict(
    gene_limit=3000,
    hvg=3000,
    epochs_stage1=10,
    epochs_stage2=12,
    batch_stage1=12,
    batch_stage2=8,
    lr_head=1e-4,
    lr_backbone=2e-6,
    weight_decay=1e-4,
    signal_len=512,
    n_v2_infer=100,
)
