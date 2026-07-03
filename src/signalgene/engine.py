"""Training and evaluation loops shared by both pipelines.

The loop inspects the model's output shape and picks the matching loss, so the
same code runs the signal model (tuple output) and the direct model (tensor output).
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .metrics import safe_pearson
from .utils import unwrap


def compute_loss(model_output, batch):
    if isinstance(model_output, tuple):
        z_hat, x_hat = model_output
        loss_signal = F.smooth_l1_loss(z_hat, batch["signal"])
        loss_expr = F.smooth_l1_loss(torch.log1p(x_hat), torch.log1p(batch["expr"]))
        return loss_signal + loss_expr
    x_hat = model_output
    return F.smooth_l1_loss(torch.log1p(x_hat), torch.log1p(batch["expr"]))


def _next_batch(it, loader):
    """Get the next batch, restarting the iterator once it's exhausted."""
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def train_one_epoch(model, main_loader, optimizer, device,
                     aux_loaders=None, aux_weights=None):
    """Train one epoch on `main_loader`, optionally mixing in auxiliary loaders
    (e.g. other bin resolutions) at fixed loss weights, cycling them as needed."""
    model.train()
    aux_loaders = aux_loaders or {}
    aux_weights = aux_weights or {}
    aux_iters = {name: iter(loader) for name, loader in aux_loaders.items()}

    total, n_batches = 0.0, 0
    pbar = tqdm(main_loader, desc="train", leave=False)
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)

        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch["fine"], batch["mid"], batch["coarse"], batch["masked"])
        loss_all = compute_loss(out, batch)

        for name, it in aux_iters.items():
            aux_batch, aux_iters[name] = _next_batch(it, aux_loaders[name])
            aux_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in aux_batch.items()}
            aux_out = model(aux_batch["fine"], aux_batch["mid"], aux_batch["coarse"], aux_batch["masked"])
            loss_all = loss_all + aux_weights.get(name, 1.0) * compute_loss(aux_out, aux_batch)

        loss_all.backward()
        torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), max_norm=1.0)
        optimizer.step()

        val = float(loss_all.item())
        if np.isfinite(val):
            total += val
            n_batches += 1
        pbar.set_postfix(loss=f"{val:.4f}")

    return total / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses, gene_corrs = [], []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch["fine"], batch["mid"], batch["coarse"], batch["masked"])
        losses.append(float(compute_loss(out, batch).item()))

        x_hat = out[1] if isinstance(out, tuple) else out
        yt = batch["expr"].cpu().numpy()
        yp = x_hat.cpu().numpy()
        for g in range(yt.shape[1]):
            c = safe_pearson(np.log1p(yt[:, g]), np.log1p(yp[:, g]))
            if not np.isnan(c):
                gene_corrs.append(c)

    return (
        float(np.mean(losses)) if losses else np.nan,
        float(np.mean(gene_corrs)) if gene_corrs else np.nan,
    )
