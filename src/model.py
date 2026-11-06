"""U-Net model and the loss function."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F


def build_unet(encoder: str = "resnet18", encoder_weights: str | None = "imagenet") -> torch.nn.Module:
    """U-Net with a ResNet encoder, 2 input channels (VV, VH) and 1 output channel (water logit).

    smp adapts the ImageNet first layer (3 RGB channels) to our 2 SAR channels.
    """
    return smp.Unet(encoder, encoder_weights=encoder_weights, in_channels=2, classes=1)


def flood_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """0.5 * BCE + 0.5 * Dice, computed only on labelled pixels (target != -1).

    logits: (B, 1, H, W) raw model output. target: (B, H, W) with -1 / 0 / 1.
    BCE gives a stable per-pixel signal; Dice directly rewards overlap and copes with
    the class imbalance (most pixels are not water).
    """
    logits = logits[:, 0].float()  # (B, H, W)
    valid = target != -1
    if not valid.any():
        return logits.sum() * 0.0  # nothing labelled: zero loss that still supports backward()
    water = (target == 1).float()

    bce = F.binary_cross_entropy_with_logits(logits[valid], water[valid])

    prob = torch.sigmoid(logits) * valid  # zero out unlabelled pixels
    inter = (prob * water).sum()
    dice = 1 - (2 * inter + 1) / (prob.sum() + (water * valid).sum() + 1)  # +1 avoids 0/0
    return 0.5 * bce + 0.5 * dice
