"""Tests for the model, loss, training loop, evaluation and figures.

These run the real U-Net code on the tiny fake dataset (2 epochs on the CPU), so they check that everything
works together and that the leakage rule holds - not that the model is accurate.
"""

import json  # reads history.json
from pathlib import Path  # file paths

import numpy as np  # arrays
import torch  # tensors

# The whole train module is imported (not just train()) so a test can replace its get_split (see leakage test).
from src import train as train_module
from src.baselines import fit_baselines
from src.evaluate import evaluate
from src.model import build_unet, flood_loss
from src.visualize import error_map, figures


def test_unet_output_shape():
    """The U-Net turns a batch of 2-band images into one value (logit) per pixel, same height and width."""
    # Random weights (no download); eval() so BatchNorm behaves deterministically.
    model = build_unet(encoder_weights=None).eval()
    # No gradients needed for a shape check.
    with torch.no_grad():
        # Input: batch of 2 random images, 2 channels, 64x64 -> output: (2, 1, 64, 64).
        assert model(torch.randn(2, 2, 64, 64)).shape == (2, 1, 64, 64)  # one logit per pixel


def test_loss_ignores_no_data_pixels():
    """Changing predictions only on -1 pixels must leave the loss exactly the same."""
    # One 2x2 label: top row water / not water, bottom row "no data".
    target = torch.tensor([[[1, 0], [-1, -1]]])  # bottom row = no data
    # Neutral predictions (logit 0 = probability 0.5) everywhere.
    logits = torch.zeros(1, 1, 2, 2)
    # Copy, then put extreme values on the bottom (no-data) row only.
    changed = logits.clone()
    changed[0, 0, 1] = 50.0  # wild predictions only on the no-data row
    # allclose = equal up to tiny floating-point differences.
    assert torch.allclose(flood_loss(logits, target), flood_loss(changed, target))


def test_loss_small_for_perfect_prediction():
    """Very confident, fully correct predictions should give a loss close to 0."""
    target = torch.tensor([[[1, 0], [1, 0]]])
    # Logit +20 (probability ~1) on water and -20 (probability ~0) elsewhere; unsqueeze(1) adds the channel axis.
    logits = (target * 40.0 - 20.0).unsqueeze(1)  # +20 on water, -20 elsewhere
    assert flood_loss(logits, target) < 1e-3


def test_train_saves_best_checkpoint(cfg):
    """train() writes a checkpoint, and the reported best epoch is the one with the highest validation IoU."""
    summary = train_module.train(cfg)
    out = Path(cfg["paths"]["output_dir"])
    # List of {"epoch", "train_loss", "val_iou"} dicts.
    history = json.loads((out / "history.json").read_text())["history"]
    assert (out / "checkpoint.pt").exists()
    # max(..., key=...) finds the epoch record with the highest val_iou.
    assert summary["best_epoch"] == max(history, key=lambda r: r["val_iou"])["epoch"]  # chosen by val IoU


def test_training_never_reads_test_or_bolivia(cfg, monkeypatch):
    """Leakage guard: record every split the training code asks for.

    If training ever looked at test or Bolivia data, the final scores on them would be too optimistic.
    monkeypatch (a built-in pytest fixture) temporarily replaces train.py's get_split with a "spy" that writes
    down the split name and then calls the real function; the original is restored after the test.
    """
    # Split names requested during training.
    asked = []
    # Keep a reference to the real function.
    real_get_split = train_module.get_split
    # The spy: append() returns None, so "None or real_get_split(...)" records the name and then returns the
    # real result.
    monkeypatch.setattr(train_module, "get_split", lambda c, split: asked.append(split) or real_get_split(c, split))
    train_module.train(cfg)
    # Training may only have asked for train and val.
    assert set(asked) == {"train", "val"}


def test_full_pipeline(cfg):
    """baselines -> train -> evaluate -> figures on the fake data (an end-to-end smoke test)."""
    fit_baselines(cfg)
    train_module.train(cfg)
    df = evaluate(cfg)
    # All 4 methods were scored on all 3 splits.
    assert set(df.method) == {"Otsu (per-chip)", "Otsu (global)", "Random Forest", "U-Net"}
    assert set(df.split) == {"val", "test", "bolivia"}
    # Every IoU is a valid number between 0 and 1.
    assert df.iou.between(0, 1).all()
    figures(cfg)
    # File names of all PNGs written to outputs/figures/.
    names = {p.name for p in (Path(cfg["paths"]["output_dir"]) / "figures").glob("*.png")}
    # Exactly the 6 expected figures.
    assert names == {"examples.png", "histogram.png", "training.png", "metrics.png",
                     "predictions_test.png", "predictions_bolivia.png"}


def test_error_map_codes():
    """Each (prediction, label) combination gets the right colour code in the prediction figures."""
    label = np.array([1, 1, 0, 0, -1])
    pred = np.array([1, 0, 1, 0, 1])
    # Codes: 2 = correct water, 4 = missed water, 3 = false alarm, 1 = correct not-water, 0 = no data.
    assert error_map(pred, label).tolist() == [2, 4, 3, 1, 0]  # hit, missed, false alarm, correct, no data
