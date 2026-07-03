"""Model definitions.

Both pipelines use the same image encoder (UNI ViT-L backbone + a small center-crop
CNN + a fusion trunk). They differ only in what sits on top of the trunk:

  - ImageToSignalGeneModel (pathway-guided): trunk -> TCN signal head -> gene decoder
  - ImageToGeneModel       (direct baseline): trunk -> gene decoder
"""

import timm
import torch
import torch.nn as nn

from .utils import safe_torch_load


class ImageEncoder(nn.Module):
    """UNI ViT-L/16 backbone (last 4 blocks fine-tuned) + center-crop CNN, fused into a 512-d trunk."""

    def __init__(self, uni_weights_path: str):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0
        )
        state_dict = safe_torch_load(uni_weights_path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.backbone.load_state_dict(state_dict, strict=False)

        for p in self.backbone.parameters():
            p.requires_grad = False
        if hasattr(self.backbone, "blocks"):
            for blk in self.backbone.blocks[-4:]:
                for p in blk.parameters():
                    p.requires_grad = True
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        self.center_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(128, 256), nn.GELU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(1024 * 3 + 256, 1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.GELU(),
        )

    def vit_pool(self, x):
        f = self.backbone.forward_features(x)
        if isinstance(f, (list, tuple)):
            f = f[-1]
        if f.ndim == 3:
            n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
            f = f[:, n_prefix:, :].mean(dim=1) if f.size(1) > n_prefix else f.mean(dim=1)
        elif f.ndim == 4:
            f = f.mean(dim=(2, 3))
        return f

    def forward(self, fine, mid, coarse, masked):
        ff = self.vit_pool(fine)
        mf = self.vit_pool(mid)
        cf = self.vit_pool(coarse)
        cc = self.center_cnn(masked)
        return self.trunk(torch.cat([ff, mf, cf, cc], dim=1))


class GeneDecoder(nn.Module):
    def __init__(self, in_dim: int, n_genes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, n_genes), nn.Softplus(),
        )

    def forward(self, h):
        return self.net(h)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation), nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation), nn.GELU(),
        )

    def forward(self, x):
        return x + self.net(x)


class SignalHeadTCN(nn.Module):
    """Projects the trunk embedding into a 1-D signal, refined by a small dilated TCN stack."""

    def __init__(self, in_dim: int, signal_dim: int, hidden: int = 256):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, 512), nn.GELU(), nn.Linear(512, signal_dim))
        self.pre = nn.Conv1d(1, hidden, 1)
        self.tcn = nn.Sequential(
            TCNBlock(hidden, 1), TCNBlock(hidden, 2), TCNBlock(hidden, 4), TCNBlock(hidden, 8)
        )
        self.out = nn.Conv1d(hidden, 1, 1)

    def forward(self, h):
        z0 = self.fc(h).unsqueeze(1)
        z = self.pre(z0)
        z = self.tcn(z)
        return self.out(z).squeeze(1)


class ImageToSignalGeneModel(nn.Module):
    """Pathway-guided pipeline: image -> trunk -> signal -> genes."""

    def __init__(self, uni_weights_path: str, signal_dim: int, n_genes: int):
        super().__init__()
        self.encoder = ImageEncoder(uni_weights_path)
        self.signal_head = SignalHeadTCN(512, signal_dim, hidden=256)
        self.gene_decoder = GeneDecoder(signal_dim, n_genes)

    def forward(self, fine, mid, coarse, masked):
        h = self.encoder(fine, mid, coarse, masked)
        z = self.signal_head(h)
        x_hat = self.gene_decoder(z)
        return z, x_hat


class ImageToGeneModel(nn.Module):
    """Baseline pipeline: image -> trunk -> genes directly, no signal bottleneck."""

    def __init__(self, uni_weights_path: str, n_genes: int):
        super().__init__()
        self.encoder = ImageEncoder(uni_weights_path)
        self.gene_decoder = GeneDecoder(512, n_genes)

    def forward(self, fine, mid, coarse, masked):
        h = self.encoder(fine, mid, coarse, masked)
        return self.gene_decoder(h)
