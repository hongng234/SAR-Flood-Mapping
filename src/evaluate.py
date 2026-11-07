"""Metrics, and the evaluation of all methods (Otsu, Random Forest, U-Net) on the val / test / Bolivia splits.

Where this file sits in the pipeline (download -> baselines -> train -> evaluate -> figures):
    - `python main.py evaluate` calls `evaluate()`, which writes outputs/metrics.csv (one row per method and split)
      and outputs/per_event.csv (one row per method, split and flood event).
    - `src/train.py` uses `Metrics` to compute the validation IoU after every epoch.
    - `src/visualize.py` uses `load_predictors()` and `iou()` for the prediction figures.

How a water map is scored (each pixel falls into one of 4 boxes; -1 "no data" pixels are skipped):
    TP (true positive)  = predicted water, really water        -> correct
    FP (false positive) = predicted water, really not water    -> false alarm
    FN (false negative) = predicted not water, really water    -> missed water
    TN (true negative)  = predicted not water, really not water -> correct
    IoU       = TP / (TP + FP + FN)   overlap / union of predicted and true water (main metric, 1.0 = perfect)
    precision = TP / (TP + FP)        of what we called water, how much really is water
    recall    = TP / (TP + FN)        of all real water, how much we found
    F1        = 2TP / (2TP + FP + FN) balance of precision and recall
"""

# Allows modern type hints on older Python versions.
from __future__ import annotations

from collections import defaultdict  # dict that creates a default value for a new key automatically
from pathlib import Path  # file paths

import numpy as np  # arrays
import pandas as pd  # tables (DataFrame) and CSV writing
import torch  # running the U-Net

# Our own helpers to list, read, clean and normalise chips (see src/data.py).
from .data import clean_db, get_split, normalize, read_chip, read_label


# --------------------------------------------------------------------------- metrics
def confusion(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Count [TP, FP, FN, TN] for the water class on one chip. Pixels with label -1 (no data) are ignored.

    Args:
        pred: (H, W) prediction, 1/True = water, 0/False = not water.
        label: (H, W) hand label with -1 / 0 / 1.
    Returns:
        int64 array of 4 numbers [TP, FP, FN, TN].
    """
    # True where a real label exists; only these pixels are counted.
    valid = label != -1
    # Predictions of the valid pixels as True/False (1-D array).
    p = pred[valid].astype(bool)
    # Truth of the valid pixels: True = water.
    t = label[valid] == 1
    # "&" = AND, "~" = NOT, np.sum counts the Trues:
    # TP = p and t, FP = p and not t, FN = not p and t, TN = not p and not t.
    return np.array([np.sum(p & t), np.sum(p & ~t), np.sum(~p & t), np.sum(~p & ~t)], dtype=np.int64)


def scores(counts: np.ndarray) -> dict:
    """Compute IoU, F1, precision and recall from [TP, FP, FN, TN].

    IoU = TP / (TP + FP + FN): overlap between predicted and true water.
    If there is no water in both prediction and label, the result counts as perfect (1.0).

    Args:
        counts: [TP, FP, FN, TN] (from confusion() or a sum of several).
    Returns:
        {"iou", "f1", "precision", "recall"} as floats between 0 and 1.
    """
    # Unpack the 4 counts; TN is not used by these metrics, so it goes into "_".
    tp, fp, fn, _ = counts

    # Small helper: safe division.
    def div(a, b):
        """Return a / b, or 1.0 when b is 0 (there was nothing to find and nothing was found = perfect)."""
        return float(a / b) if b > 0 else 1.0  # 0/0 -> 1.0 (nothing to find, nothing found)

    return {"iou": div(tp, tp + fp + fn), "f1": div(2 * tp, 2 * tp + fp + fn),
            "precision": div(tp, tp + fp), "recall": div(tp, tp + fn)}


class Metrics:
    """Adds up confusion counts over many chips: in total (micro average) and per flood event.

    Usage: m = Metrics(); m.update(pred, label, chip_id) for every chip; then m.scores() for the total
    and scores(m.per_event["Bolivia"]) for one event.
    """

    def __init__(self):
        """Start with all counts at zero."""
        # Running [TP, FP, FN, TN] over all chips.
        self.total = np.zeros(4, dtype=np.int64)
        # Event name -> running [TP, FP, FN, TN]; a new event automatically starts at four zeros.
        self.per_event = defaultdict(lambda: np.zeros(4, dtype=np.int64))

    def update(self, pred: np.ndarray, label: np.ndarray, chip_id: str) -> None:
        """Add the counts of one chip to the total and to its event."""
        # [TP, FP, FN, TN] of this chip.
        c = confusion(pred, label)
        # Add to the overall total.
        self.total += c
        # Add to the event the chip belongs to (the part of the id before "_").
        self.per_event[chip_id.split("_")[0]] += c  # "Bolivia_103757" -> "Bolivia"

    def scores(self) -> dict:
        """Scores of all chips together."""
        # Micro average: counts are summed first, then the metric is computed once,
        # so chips with more water pixels count more than chips with little water.
        return scores(self.total)


def iou(pred: np.ndarray, label: np.ndarray) -> float:
    """IoU of a single chip (shortcut used for the titles in the prediction figures)."""
    return scores(confusion(pred, label))["iou"]


# --------------------------------------------------------------------------- predictors
def load_predictors(cfg: dict) -> dict:
    """Load every trained method and wrap each one as a function: dB image (2, H, W) -> water mask (H, W) with 0/1.

    Having the same "image in, mask out" shape for all 4 methods lets evaluate() and the figures treat them alike.

    Returns:
        {"Otsu (per-chip)": f, "Otsu (global)": f, "Random Forest": f, "U-Net": f}.
    """
    # Imported here (not at the top of the file) to avoid a circular import: train.py imports this file.
    from .baselines import load_baselines, predict_otsu_per_chip, predict_rf, predict_threshold
    from .train import get_device, load_model

    # Saved global Otsu threshold (dB) and Random Forest from outputs/baselines/.
    threshold, rf = load_baselines(cfg)
    # GPU if available.
    device = get_device()
    # Best U-Net checkpoint and the train normalisation statistics stored with it.
    model, stats = load_model(cfg, device)

    # Inner function that "remembers" model, stats and device (a closure). No gradients needed for prediction.
    @torch.no_grad()
    def predict_unet(img: np.ndarray) -> np.ndarray:
        """Normalise the chip, run the U-Net and threshold the probability at 0.5."""
        # normalize -> (2, H, W); [None] adds the batch axis -> (1, 2, H, W); then move it to the device.
        x = torch.from_numpy(normalize(img, stats)[None]).to(device)  # [None] adds the batch axis
        # Output (1, 1, H, W) -> [0, 0] takes the single map (H, W); sigmoid -> probability; > 0.5 -> water.
        return (torch.sigmoid(model(x))[0, 0] > 0.5).cpu().numpy().astype(np.uint8)

    return {
        "Otsu (per-chip)": predict_otsu_per_chip,  # needs no training, already the right shape
        # lambda = a tiny unnamed function; it fixes the learned threshold / forest as the second input.
        "Otsu (global)": lambda img: predict_threshold(img, threshold),
        "Random Forest": lambda img: predict_rf(rf, img),
        "U-Net": predict_unet,
    }


def evaluate(cfg: dict, splits=("val", "test", "bolivia")) -> pd.DataFrame:
    """Score every method on every split. Writes outputs/metrics.csv and outputs/per_event.csv.

    Args:
        cfg: config dict. splits: which splits to score.
    Returns:
        DataFrame with columns method, split, iou, f1, precision, recall (same as metrics.csv).
    """
    # The 4 methods as image -> mask functions.
    predictors = load_predictors(cfg)
    # rows -> metrics.csv, event_rows -> per_event.csv.
    rows, event_rows = [], []
    # Loop over the splits (val, test, bolivia).
    for split in splits:
        # A fresh counter for every method in this split (dict comprehension: {key: value for ...}).
        metrics = {name: Metrics() for name in predictors}
        # Loop over every chip of the split; each chip is read only once and given to all methods.
        for s in get_split(cfg, split):
            # Cleaned image (2, H, W) in dB and the label (H, W).
            img, label = clean_db(read_chip(s.image_path)), read_label(s.label_path)
            # Run each method on the chip and add its counts.
            for name, predict in predictors.items():
                metrics[name].update(predict(img), label, s.chip_id)
        # After all chips: turn the counts into one result row per method.
        for name, m in metrics.items():
            # "**" unpacks the scores dict into this row: {"method":..., "split":..., "iou":..., "f1":..., ...}.
            rows.append({"method": name, "split": split, **m.scores()})
            # One extra row per flood event inside this split (list comprehension added to event_rows).
            event_rows += [{"method": name, "split": split, "event": ev, **scores(c)} for ev, c in m.per_event.items()]

    # Output folder.
    out = Path(cfg["paths"]["output_dir"])
    # Turn the list of dicts into a table.
    df = pd.DataFrame(rows)
    # Save both tables as CSV with 4 decimals (index=False: do not write the row numbers).
    df.to_csv(out / "metrics.csv", index=False, float_format="%.4f")
    pd.DataFrame(event_rows).to_csv(out / "per_event.csv", index=False, float_format="%.4f")
    return df
