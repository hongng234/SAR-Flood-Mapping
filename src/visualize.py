"""Figures saved to outputs/figures/: data examples, histogram, training curve, metrics and predictions.

Where this file sits in the pipeline (download -> baselines -> train -> evaluate -> figures):
    `python main.py figures` calls `figures()`, which needs the outputs of the earlier steps
    (otsu.json, history.json, metrics.csv, checkpoint.pt) and draws 6 PNG files:
        examples.png            VV, VH and label of 4 chips from different flood events
        histogram.png           brightness of water vs. non-water pixels, with the Otsu threshold
        training.png            training loss and validation IoU per epoch
        metrics.png             IoU of every method on val / test / Bolivia
        predictions_test.png    error maps of every method on a few test chips
        predictions_bolivia.png the same on the unseen Bolivia flood
"""

# Allows modern type hints on older Python versions.
from __future__ import annotations

import json  # reads otsu.json and history.json
from pathlib import Path  # file paths

import matplotlib  # plotting library

# Use the "Agg" backend: draw straight into PNG files without opening a window (works on servers and in tests).
# It must be chosen BEFORE pyplot is imported, which is why the imports below carry "noqa: E402"
# (it tells the linter ruff that an import placed after code is intentional).
matplotlib.use("Agg")  # draw straight to PNG files, no window
import matplotlib.pyplot as plt  # noqa: E402  # the main plotting interface
import numpy as np  # noqa: E402  # arrays
import pandas as pd  # noqa: E402  # reads metrics.csv and the training history
from matplotlib.colors import ListedColormap  # noqa: E402  # maps the integers 0,1,2,... to fixed colours
from matplotlib.patches import Patch  # noqa: E402  # coloured squares for hand-made legends

# Our own helpers to list, read and clean chips (see src/data.py).
from .data import clean_db, get_split, read_chip, read_label  # noqa: E402

# One fixed colour per method, so a method always has the same colour in every figure.
COLORS = {"Otsu (per-chip)": "#2a78d6", "Otsu (global)": "#eb6834", "Random Forest": "#1baf7a", "U-Net": "#6250d6"}
# Named colours (hex codes) used for maps and lines.
GREY, LIGHT, BLUE, ORANGE, PURPLE = "#bdbcb6", "#f2f1ed", "#2a78d6", "#eb6834", "#6250d6"
# Colour map for labels. Labels are -1/0/1, so we draw label + 1 = 0/1/2 -> grey / light / blue.
LABEL_CMAP = ListedColormap([GREY, LIGHT, BLUE])  # label + 1: 0 no data, 1 not water, 2 water
# Colour map for the error map codes 0..4 produced by error_map().
ERROR_CMAP = ListedColormap([GREY, LIGHT, BLUE, ORANGE, PURPLE])  # see error_map()
# (text, colour) pairs for the legend under the prediction figures, in the same order as the codes.
ERROR_LEGEND = [("No data", GREY), ("Correct: not water", LIGHT), ("Correct: water", BLUE),
                ("False alarm", ORANGE), ("Missed water", PURPLE)]
# Global style for every figure: no top/right border lines, a light grid drawn behind the data, no legend box.
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "axes.axisbelow": True, "legend.frameon": False})


def _save(fig, cfg: dict, name: str) -> None:
    """Save a figure as outputs/figures/<name> and close it to free memory. (Leading "_" = internal helper.)"""
    # outputs/figures/ (created if missing).
    folder = Path(cfg["paths"]["output_dir"]) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    # dpi = resolution; bbox_inches="tight" trims empty white margins.
    fig.savefig(folder / name, dpi=120, bbox_inches="tight")
    # Close the figure, otherwise matplotlib keeps every figure in memory.
    plt.close(fig)
    print(f"saved figures/{name}")


def _legend(fig, items) -> None:
    """Draw a colour legend under the figure (thin outline so light colours stay visible).

    Args:
        fig: the matplotlib figure. items: list of (text, colour) pairs.
    """
    # One coloured square (Patch) per item, laid out in a single row at the bottom centre.
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#52514e", lw=0.5, label=t) for t, c in items],
               loc="lower center", ncol=len(items))


def error_map(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Give each pixel a code that says whether the prediction is right or wrong, and how.

    Codes: 0 no data, 1 correct not-water, 2 correct water, 3 false alarm, 4 missed water.

    Args:
        pred: (H, W) prediction 0/1. label: (H, W) hand label -1/0/1.
    Returns:
        (H, W) uint8 array of codes, drawn with ERROR_CMAP.
    """
    # Start with 1 ("correct: not water") everywhere, then overwrite the other cases.
    out = np.ones(label.shape, dtype=np.uint8)
    # Water in the label AND predicted water -> correct water.
    out[(label == 1) & (pred == 1)] = 2
    # Not water in the label but predicted water -> false alarm.
    out[(label == 0) & (pred == 1)] = 3
    # Water in the label but predicted not water -> missed water.
    out[(label == 1) & (pred == 0)] = 4
    # No-data pixels last, so they win over any code set above.
    out[label == -1] = 0
    return out


def plot_examples(cfg: dict) -> None:
    """Draw VV, VH and the label for one train chip from each of 4 different flood events -> examples.png."""
    # picks = chosen chips; seen = events already used (a set gives fast "is it in there?" checks).
    picks, seen = [], set()
    # Walk through the training chips and take the first chip of each new event.
    for s in get_split(cfg, "train"):
        if s.event not in seen:
            seen.add(s.event)
            picks.append(s)
    # Keep at most 4 chips (4 rows).
    picks = picks[:4]
    # A grid: one row per chip, 3 columns (VV, VH, label). squeeze=False keeps `axes` 2-D even with 1 row.
    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)), squeeze=False)
    # Fill each row with one chip.
    for row, s in zip(axes, picks):
        # Cleaned (2, H, W) image in dB.
        img = clean_db(read_chip(s.image_path))
        # Show VV in grey. vmin/vmax fix the colour scale: -30 dB = black, 0 dB = white.
        row[0].imshow(img[0], cmap="gray", vmin=-30, vmax=0)  # same dB stretch for every chip
        # Same for VH.
        row[1].imshow(img[1], cmap="gray", vmin=-30, vmax=0)
        # Label + 1 -> 0/1/2 coloured with LABEL_CMAP; "nearest" keeps sharp pixel edges (no blurring).
        row[2].imshow(read_label(s.label_path) + 1, cmap=LABEL_CMAP, vmin=0, vmax=2, interpolation="nearest")
        # Give each of the 3 panels its title and hide the axis ticks.
        for ax, title in zip(row, [f"{s.chip_id} VV (dB)", "VH (dB)", "Hand label"]):
            ax.set_title(title)
            ax.axis("off")
    # Arrange panels neatly, leaving 3 % space at the bottom for the legend.
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _legend(fig, [("No data", GREY), ("Not water", LIGHT), ("Water", BLUE)])
    _save(fig, cfg, "examples.png")


def plot_histogram(cfg: dict) -> None:
    """Histogram of VV brightness for water vs. non-water train pixels, with the global Otsu threshold.

    Shows why thresholding works (water is darker) and why it is not perfect (the two curves overlap).
    Saves histogram.png.
    """
    # Fixed seed so the sampled pixels are the same every run.
    rng = np.random.default_rng(0)
    # Class (0 = not water, 1 = water) -> list of sampled VV values per chip.
    values = {0: [], 1: []}
    # Loop over the training chips.
    for s in get_split(cfg, "train"):
        # VV band (H, W) and label (H, W).
        vv, label = clean_db(read_chip(s.image_path))[0], read_label(s.label_path)
        # For each class, take up to 5000 random pixels of that class.
        for cls in (0, 1):
            # 1-D array of the VV values of pixels with this label.
            v = vv[label == cls]
            values[cls].append(rng.choice(v, min(5000, v.size), replace=False))  # sample to keep it fast
    # One plot, 7 x 3.6 inches.
    fig, ax = plt.subplots(figsize=(7, 3.6))
    # 91 bin edges from -40 dB to +5 dB (= 90 bins of 0.5 dB).
    bins = np.linspace(-40, 5, 91)
    # density=True scales each curve to area 1, so both classes are comparable despite different pixel counts.
    # histtype="step" draws an outline instead of filled bars.
    ax.hist(np.concatenate(values[0]), bins, density=True, histtype="step", lw=2, color=ORANGE, label="Not water")
    ax.hist(np.concatenate(values[1]), bins, density=True, histtype="step", lw=2, color=BLUE, label="Water")
    # Where the baselines step saved the global Otsu threshold.
    otsu_file = Path(cfg["paths"]["output_dir"]) / "baselines" / "otsu.json"
    # Only draw the threshold line if the baselines step has been run.
    if otsu_file.exists():
        t = json.loads(otsu_file.read_text())["threshold_db"]
        # Dashed vertical line at the threshold, with a text label near the top of the plot.
        ax.axvline(t, color="#52514e", ls="--")
        ax.text(t + 0.5, ax.get_ylim()[1] * 0.9, f"Otsu threshold {t:.1f} dB")
    ax.set(title="VV backscatter, train pixels", xlabel="dB", ylabel="density")
    ax.legend(loc="upper left")
    _save(fig, cfg, "histogram.png")


def plot_training(cfg: dict) -> None:
    """Training loss and validation IoU per epoch; the dashed line marks the saved (best) epoch -> training.png."""
    # history.json written by train(): {"summary": {...}, "history": [{"epoch", "train_loss", "val_iou"}, ...]}.
    data = json.loads((Path(cfg["paths"]["output_dir"]) / "history.json").read_text())
    # Table with one row per epoch; columns can be read as h.epoch, h.train_loss, h.val_iou.
    h = pd.DataFrame(data["history"])
    # Two plots side by side: a1 = loss, a2 = validation IoU.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    # Loss should go down as the model learns.
    a1.plot(h.epoch, h.train_loss, lw=2, color=BLUE)
    a1.set(title="Training loss (BCE + Dice)", xlabel="epoch")
    # Validation IoU should go up, then level off.
    a2.plot(h.epoch, h.val_iou, lw=2, color=PURPLE)
    # Dashed line at the epoch whose weights were saved.
    a2.axvline(data["summary"]["best_epoch"], color="#52514e", ls="--")
    a2.set(title=f"Validation IoU (best {h.val_iou.max():.3f})", xlabel="epoch")
    _save(fig, cfg, "training.png")


def plot_metrics(cfg: dict) -> None:
    """Grouped bar chart: IoU of each method on val, test and Bolivia -> metrics.png."""
    # Results table written by evaluate().
    df = pd.read_csv(Path(cfg["paths"]["output_dir"]) / "metrics.csv")
    # Order of the groups on the x axis.
    splits = ["val", "test", "bolivia"]
    # x = centre position of each group (0, 1, 2); w = width of one bar.
    x, w = np.arange(len(splits)), 0.2
    fig, ax = plt.subplots(figsize=(8, 3.8))
    # One set of bars per method; i = 0..3 is used to shift the bars side by side; enumerate gives (i, item).
    for i, (method, color) in enumerate(COLORS.items()):
        # IoU of this method on each split: filter the rows, take the iou column, first (only) value.
        vals = [df[(df.method == method) & (df.split == s)].iou.iloc[0] for s in splits]
        # Offsets -1.5w, -0.5w, +0.5w, +1.5w centre the 4 bars on x; 0.92 leaves a small gap between bars.
        bars = ax.bar(x + (i - 1.5) * w, vals, w * 0.92, color=color, label=method)  # 4 bars side by side
        # Write the value (2 decimals) on top of each bar.
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    # Group names under the x axis.
    ax.set_xticks(x, ["Validation", "Test", "Bolivia (unseen event)"])
    # IoU is between 0 and 1.
    ax.set(ylim=(0, 1), ylabel="IoU (water)")
    # Only horizontal grid lines.
    ax.grid(axis="x", visible=False)
    # Legend in one row above the plot.
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    _save(fig, cfg, "metrics.png")


def plot_predictions(cfg: dict, split: str, n: int = 4) -> None:
    """For n chips of a split: VV image, hand label, and an error map for each method with its IoU.

    Saves predictions_<split>.png. Each row is one chip; each column after the first two is one method.

    Args:
        cfg: config dict. split: "test" or "bolivia" (any split works). n: number of chips (rows).
    """
    # Imported here to avoid loading the model code unless this figure is drawn.
    from .evaluate import iou, load_predictors

    # The 4 trained methods as image -> mask functions.
    predictors = load_predictors(cfg)
    # Chips with at least 5 % water: first one chip per event, then fill up (Bolivia has only one event).
    # (read_label(...) == 1).mean() = fraction of pixels that are water.
    wet = [s for s in get_split(cfg, split) if (read_label(s.label_path) == 1).mean() > 0.05]
    # First pass: one chip per flood event, for variety.
    picks, seen = [], set()
    for s in wet:
        if s.event not in seen:
            seen.add(s.event)
            picks.append(s)
    # Second pass: add the remaining wet chips in order, then keep the first n.
    picks = (picks + [s for s in wet if s not in picks])[:n]
    # Number of columns: VV + label + one per method.
    cols = 2 + len(predictors)
    # squeeze=False keeps `axes` 2-D (rows x cols) even with a single row.
    fig, axes = plt.subplots(len(picks), cols, figsize=(2.6 * cols, 2.7 * len(picks)), squeeze=False)
    # Fill one row per chip.
    for row, s in zip(axes, picks):
        # Cleaned image and label of this chip.
        img, label = clean_db(read_chip(s.image_path)), read_label(s.label_path)
        # Column 0: the VV image in grey.
        row[0].imshow(img[0], cmap="gray", vmin=-30, vmax=0)
        row[0].set_title(f"{s.chip_id}\nVV (dB)")
        # Column 1: the hand label.
        row[1].imshow(label + 1, cmap=LABEL_CMAP, vmin=0, vmax=2, interpolation="nearest")
        row[1].set_title("Hand label")
        # Columns 2..: one error map per method, titled with the method name and its IoU on this chip.
        for ax, (name, predict) in zip(row[2:], predictors.items()):
            pred = predict(img)
            ax.imshow(error_map(pred, label), cmap=ERROR_CMAP, vmin=0, vmax=4, interpolation="nearest")
            ax.set_title(f"{name}\nIoU {iou(pred, label):.2f}")
        # Hide ticks on every panel of the row.
        for ax in row:
            ax.axis("off")
    # Leave 4 % space at the bottom for the legend.
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _legend(fig, ERROR_LEGEND)
    _save(fig, cfg, f"predictions_{split}.png")


def figures(cfg: dict) -> None:
    """Draw all 6 figures (called by `python main.py figures`)."""
    plot_examples(cfg)  # examples.png
    plot_histogram(cfg)  # histogram.png
    plot_training(cfg)  # training.png
    plot_metrics(cfg)  # metrics.png
    plot_predictions(cfg, "test")  # predictions_test.png
    plot_predictions(cfg, "bolivia")  # predictions_bolivia.png
