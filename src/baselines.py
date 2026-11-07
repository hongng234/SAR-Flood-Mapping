"""Classical (non-deep-learning) baselines: Otsu thresholding and a per-pixel Random Forest.

Where this file sits in the pipeline (download -> baselines -> train -> evaluate -> figures):
    - `python main.py baselines` calls `fit_baselines()`, which learns the global Otsu threshold and the Random
      Forest from the TRAIN split and saves them to outputs/baselines/.
    - `src/evaluate.py` calls `load_baselines()` and the predict_* functions to score them on val / test / Bolivia.

Why baselines: they show how much the U-Net really adds compared to simple, well-known methods.
    - Otsu: the classic flood-mapping method. Water is dark in SAR, so "everything darker than X dB is water".
      Otsu's algorithm picks X automatically from the brightness histogram.
    - Random Forest: a standard machine-learning classifier that decides pixel by pixel from a few features.

Both are fitted on the TRAIN split only (no leakage from val/test/Bolivia).
"""

# Lets us write modern type hints (e.g. tuple[float, ...]) on older Python versions.
from __future__ import annotations

import json  # saves/loads the Otsu threshold as a tiny JSON file
import pickle  # saves/loads the trained Random Forest object to a file
from pathlib import Path  # file paths

import numpy as np  # arrays
from scipy.ndimage import median_filter  # smoothing filter that reduces radar speckle noise
from skimage.filters import threshold_otsu  # Otsu's automatic threshold algorithm
from sklearn.ensemble import RandomForestClassifier  # the Random Forest model

# Our own helpers (see src/data.py): list a split, read image/label, clean bad values.
from .data import clean_db, get_split, read_chip, read_label


# --------------------------------------------------------------------------- Otsu
def predict_threshold(img: np.ndarray, threshold: float) -> np.ndarray:
    """Turn a SAR chip into a water map with a single brightness cut-off.

    Args:
        img: (2, H, W) float32 chip; band 0 = VV, band 1 = VH, both in dB.
        threshold: cut-off in dB; pixels darker (lower) than this are called water.
    Returns:
        (H, W) uint8 mask, 1 = water, 0 = not water.
    Why: calm water reflects radar away from the satellite like a mirror, so it looks dark.
    """
    # img[0] is the VV band; "<" gives True for dark pixels; astype turns True/False into 1/0.
    return (img[0] < threshold).astype(np.uint8)


def predict_otsu_per_chip(img: np.ndarray) -> np.ndarray:
    """Unsupervised method: compute a separate Otsu threshold from each chip's own VV histogram.

    Otsu looks at the histogram (count of pixels per brightness level) and finds the cut that best separates
    it into two groups ("dark" and "bright"). Nothing is learned from training data.
    Weakness: on a chip with almost no water, Otsu still splits it in two, so it calls dark land "water".

    Args:
        img: (2, H, W) cleaned chip in dB.
    Returns:
        (H, W) uint8 water mask.
    """
    # Only the VV band is used.
    vv = img[0]
    # If every pixel has the same value there is no histogram to split, so Otsu would fail.
    if vv.min() == vv.max():  # constant chip -> Otsu is undefined, predict no water
        return np.zeros(vv.shape, dtype=np.uint8)
    # Otherwise compute this chip's own threshold and apply it.
    return predict_threshold(img, threshold_otsu(vv))


def fit_global_otsu(cfg: dict, pixels_per_chip: int = 20000) -> float:
    """Learn ONE threshold for all chips: run Otsu on a random sample of labelled VV pixels from the TRAIN split.

    Sampling (instead of using all ~66 million pixels) keeps it fast; every chip contributes equally.

    Args:
        cfg: config dict (needs seed and the data paths).
        pixels_per_chip: how many random pixels to take from each train chip.
    Returns:
        The threshold in dB (about -12.45 dB on the real data).
    """
    # Seeded random generator -> the same pixels are sampled every run (reproducible).
    rng = np.random.default_rng(cfg["seed"])
    # List of 1-D arrays, one per chip, with the sampled VV values.
    values = []
    # Loop over every training chip.
    for s in get_split(cfg, "train"):
        # Read + clean the chip and keep only band 0 (VV), shape (H, W).
        vv = clean_db(read_chip(s.image_path))[0]
        # Boolean indexing: keep only pixels whose label is not -1 (i.e. labelled pixels). Result is 1-D.
        v = vv[read_label(s.label_path) != -1]  # only labelled pixels
        # Pick up to `pixels_per_chip` of them at random, without picking the same pixel twice.
        values.append(rng.choice(v, min(pixels_per_chip, v.size), replace=False))
    # Join all samples into one long array and let Otsu find the best cut; float() makes it a plain Python number.
    return float(threshold_otsu(np.concatenate(values)))


# --------------------------------------------------------------------------- Random Forest
def pixel_features(img: np.ndarray) -> np.ndarray:
    """Build 5 numbers ("features") per pixel for the Random Forest.

    Features: VV, VH, VV-VH, and 3x3 median-filtered VV and VH.
      - VV-VH in dB is the polarisation ratio (a log ratio becomes a difference); it differs between water,
        vegetation and buildings.
      - The median filter replaces each pixel with the median of its 3x3 neighbourhood, which reduces speckle
        noise and gives the model a little information about the neighbours.

    Args:
        img: (2, H, W) cleaned chip in dB.
    Returns:
        (H, W, 5) float32 array: for every pixel, its 5 feature values.
    """
    # Split the 2 bands into separate (H, W) arrays.
    vv, vh = img[0], img[1]
    # The 5 feature images, each (H, W).
    feats = [vv, vh, vv - vh, median_filter(vv, size=3), median_filter(vh, size=3)]
    # Stack them on a new last axis -> (H, W, 5).
    return np.stack(feats, axis=-1).astype(np.float32)


def fit_random_forest(cfg: dict) -> RandomForestClassifier:
    """Train a Random Forest on random labelled pixels, the same number from every TRAIN chip.

    A Random Forest is many decision trees, each trained on a random subset of the data; they vote per pixel.
    Using a sample of pixels (cfg baselines.rf_pixels in total) keeps training to a few seconds.

    Args:
        cfg: config dict (seed, baselines.rf_pixels / rf_trees / rf_max_depth, data paths).
    Returns:
        The fitted RandomForestClassifier.
    """
    # Seeded random generator for reproducible pixel sampling.
    rng = np.random.default_rng(cfg["seed"])
    # All training chips.
    samples = get_split(cfg, "train")
    # Pixels to take per chip so the total is about rf_pixels ("//" is integer division).
    per_chip = cfg["baselines"]["rf_pixels"] // len(samples)
    # xs = feature rows, ys = matching labels; one entry per chip, joined at the end.
    xs, ys = [], []
    # Loop over every training chip and collect its sampled pixels.
    for s in samples:
        # (H, W, 5) features reshaped to (H*W, 5): one row per pixel, 5 columns. -1 means "work this size out".
        feats = pixel_features(clean_db(read_chip(s.image_path))).reshape(-1, 5)  # one row per pixel
        # Label flattened to (H*W,) so row i of feats matches entry i of label.
        label = read_label(s.label_path).ravel()
        # Positions of pixels with a real label (0 or 1); -1 "no data" pixels are never used for training.
        valid = np.flatnonzero(label != -1)  # indices of labelled pixels
        # Choose up to per_chip of these positions at random, without repeats.
        pick = rng.choice(valid, min(per_chip, valid.size), replace=False)
        # Keep the chosen rows of features and their labels.
        xs.append(feats[pick])
        ys.append(label[pick])
    # Create the model: n_estimators = number of trees, max_depth limits tree size (less overfitting),
    # n_jobs=-1 uses all CPU cores, random_state makes the result reproducible.
    rf = RandomForestClassifier(n_estimators=cfg["baselines"]["rf_trees"], max_depth=cfg["baselines"]["rf_max_depth"],
                                n_jobs=-1, random_state=cfg["seed"])
    # Train on all sampled pixels from all chips at once.
    rf.fit(np.concatenate(xs), np.concatenate(ys))
    return rf


def predict_rf(rf: RandomForestClassifier, img: np.ndarray) -> np.ndarray:
    """Classify every pixel of a chip with the Random Forest and reshape the answers back into an image.

    Args:
        rf: fitted Random Forest. img: (2, H, W) cleaned chip in dB.
    Returns:
        (H, W) uint8 water mask.
    """
    # (H, W, 5) features for this chip.
    feats = pixel_features(img)
    # Predict one label per row (pixel), then reshape the flat answer to (H, W) = img.shape[1:].
    return rf.predict(feats.reshape(-1, 5)).reshape(img.shape[1:]).astype(np.uint8)


# --------------------------------------------------------------------------- save / load
def fit_baselines(cfg: dict) -> None:
    """Fit both baselines on the train split and save them to outputs/baselines/ (otsu.json and rf.pkl).

    Called by `python main.py baselines`.
    """
    # Output folder outputs/baselines/ (created if missing).
    out = Path(cfg["paths"]["output_dir"]) / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    # 1) Global Otsu threshold from train pixels.
    threshold = fit_global_otsu(cfg)
    print(f"global Otsu threshold: {threshold:.2f} dB")
    # Save it as JSON: {"threshold_db": -12.45...}.
    (out / "otsu.json").write_text(json.dumps({"threshold_db": threshold}))
    # 2) Random Forest from train pixels.
    rf = fit_random_forest(cfg)
    # pickle writes the whole Python object to a binary file ("wb" = write binary).
    with open(out / "rf.pkl", "wb") as f:
        pickle.dump(rf, f)
    print("random forest saved")


def load_baselines(cfg: dict) -> tuple[float, RandomForestClassifier]:
    """Load the saved Otsu threshold and Random Forest (written by fit_baselines).

    Unpickling can run code, so only load files you trust; these are our own files, so it is safe.

    Returns:
        (threshold in dB, fitted Random Forest).
    """
    # Folder where fit_baselines saved the files.
    out = Path(cfg["paths"]["output_dir"]) / "baselines"
    # Read the JSON text, parse it into a dict and take the threshold value.
    threshold = json.loads((out / "otsu.json").read_text())["threshold_db"]
    # "rb" = read binary; pickle.load rebuilds the Random Forest object.
    with open(out / "rf.pkl", "rb") as f:
        return threshold, pickle.load(f)
