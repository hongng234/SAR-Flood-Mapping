"""The deep-learning model (U-Net) and the loss function used to train it.

Where this file sits in the pipeline (download -> baselines -> train -> evaluate -> figures):
    - `src/train.py` builds the model with `build_unet()` and trains it by minimising `flood_loss()`.
    - `src/train.py:load_model` rebuilds the same model to load the saved weights for evaluation.

What a U-Net is (in plain words):
    - An "encoder" shrinks the image step by step (512 -> 256 -> ... -> 16 pixels) while learning what is in it.
      Here the encoder is ResNet18, a well-known image network pre-trained on ImageNet (millions of photos).
    - A "decoder" grows it back to full size to output one value per pixel.
    - "Skip connections" copy detail from each encoder level to the matching decoder level so edges stay sharp.
    The drawing of these steps looks like the letter U, hence the name.
"""

# Lets us write `str | None` type hints on older Python versions.
from __future__ import annotations

import segmentation_models_pytorch as smp  # library with ready-made segmentation models (U-Net etc.)
import torch  # PyTorch tensors and neural-network modules
import torch.nn.functional as F  # PyTorch functions such as the BCE loss


def build_unet(encoder: str = "resnet18", encoder_weights: str | None = "imagenet") -> torch.nn.Module:
    """Create a U-Net with a ResNet encoder, 2 input channels (VV, VH) and 1 output channel (water logit).

    A "logit" is the raw model output for a pixel: any real number. sigmoid(logit) turns it into a
    probability between 0 and 1 that the pixel is water.
    smp adapts the ImageNet first layer (made for 3 RGB channels) to our 2 SAR channels.

    Args:
        encoder: name of the encoder network (config.yaml model.encoder).
        encoder_weights: "imagenet" to start from pre-trained weights, or None for random weights (used in tests
            and when loading a checkpoint, because the weights then come from the file).
    Returns:
        The U-Net as a torch.nn.Module. Input (B, 2, H, W) -> output (B, 1, H, W) logits.
    """
    # in_channels=2: VV and VH. classes=1: one output map (water vs. not water).
    return smp.Unet(encoder, encoder_weights=encoder_weights, in_channels=2, classes=1)


def flood_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Loss = 0.5 * BCE + 0.5 * Dice, computed only on labelled pixels (target != -1).

    The loss is the number training tries to make small; lower = predictions closer to the labels.
      - BCE (binary cross-entropy): per-pixel penalty for being confidently wrong; gives a stable signal.
      - Dice: 1 - overlap between predicted and true water. It directly rewards overlap and copes with
        class imbalance (most pixels are not water, so BCE alone could learn "never water" and still look good).

    Args:
        logits: (B, 1, H, W) raw model output (B = batch size, number of chips at once).
        target: (B, H, W) labels with -1 (no data, ignored) / 0 / 1.
    Returns:
        Scalar tensor (one number) that supports .backward().
    """
    # Drop the channel axis -> (B, H, W); .float() makes sure we compute in float32 even under mixed precision.
    logits = logits[:, 0].float()  # (B, H, W)
    # Boolean mask: True where the pixel has a real label (0 or 1).
    valid = target != -1
    # Edge case: a batch with no labelled pixel at all.
    if not valid.any():
        # Return 0, but built from logits so PyTorch can still run backward() without an error.
        return logits.sum() * 0.0  # nothing labelled: zero loss that still supports backward()
    # 1.0 where the label is water, 0.0 elsewhere (float because the losses need floats).
    water = (target == 1).float()

    # BCE over valid pixels only: logits[valid] picks the labelled pixels as a 1-D list.
    # "with_logits" applies the sigmoid internally in a numerically safe way.
    bce = F.binary_cross_entropy_with_logits(logits[valid], water[valid])

    # Probabilities 0..1; multiplying by `valid` (True=1, False=0) sets unlabelled pixels to 0.
    prob = torch.sigmoid(logits) * valid  # zero out unlabelled pixels
    # "Intersection": how much predicted water overlaps true water (soft version, using probabilities).
    inter = (prob * water).sum()
    # Soft Dice = 2*overlap / (predicted + true). Loss is 1 - Dice. The +1 terms avoid 0/0 on chips without water.
    dice = 1 - (2 * inter + 1) / (prob.sum() + (water * valid).sum() + 1)  # +1 avoids 0/0
    # Equal mix of the two losses.
    return 0.5 * bce + 0.5 * dice
