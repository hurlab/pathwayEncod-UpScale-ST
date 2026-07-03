"""Small shared helpers: seeding, checkpoint loading, DataParallel wrapping."""

import random
from typing import List

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_torch_load(path: str, map_location: str = "cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def wrap_model(model: nn.Module, gpu_ids: List[int], primary: torch.device) -> nn.Module:
    """Move model to the primary device and wrap with DataParallel if more than one GPU is given."""
    model = model.to(primary)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    return model


def unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module regardless of DataParallel wrapping."""
    return model.module if isinstance(model, nn.DataParallel) else model
