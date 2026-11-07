"""Shared test fixtures: a tiny fake dataset with the same folders and file names as Sen1Floods11.

Each fake chip is a dark disc (water) on brighter land, saved as a real GeoTIFF, so the real
reading / training / evaluation code runs without downloading anything.

pytest loads this file automatically before the tests. A "fixture" is a function decorated with
@pytest.fixture; a test that lists the fixture's name as a parameter (e.g. `def test_x(cfg):`) receives
its return value. Here:
    fake_data -> folder with the fake dataset (written once per test run)
    cfg       -> the real config.yaml, redirected to the fake data and made small so tests run fast on a CPU
"""

# Allows modern type hints on older Python versions.
from __future__ import annotations

from pathlib import Path  # file paths

import numpy as np  # builds the fake images
import pytest  # test framework (fixtures)
import rasterio  # writes the fake images as GeoTIFF files, like the real data
from rasterio.transform import from_origin  # gives the fake GeoTIFFs a simple map position

# The real config loader, so tests use the real config.yaml as a starting point.
from main import load_config

# Side length of a fake chip in pixels; small so tests are fast.
SIZE = 64  # fake chip size (real chips are 512 x 512)


def make_chip(rng: np.random.Generator, size: int = SIZE):
    """Build one fake SAR chip and its label: a dark round "lake" at a random place on brighter land.

    Args:
        rng: numpy random generator (seeded by the caller, so chips are reproducible).
        size: side length in pixels.
    Returns:
        (image (2, H, W) float32 in dB with a few NaN, label (H, W) int16 with -1 / 0 / 1).
    """
    # Two (size, size) grids holding each pixel's row (yy) and column (xx) number.
    yy, xx = np.mgrid[:size, :size]
    # Random centre of the disc, kept away from the borders (between 1/4 and 3/4 of the size).
    cy, cx = rng.integers(size // 4, 3 * size // 4, size=2)
    # True inside a circle of random radius: (distance from centre)^2 < radius^2.
    water = (yy - cy) ** 2 + (xx - cx) ** 2 < rng.integers(size // 6, size // 3) ** 2
    # VV: -22 dB on water, -8 dB on land, plus random noise (std 1.5 dB) to imitate speckle.
    vv = np.where(water, -22.0, -8.0) + rng.normal(0, 1.5, (size, size))  # water dark, land bright
    # VH: same idea, VH is always darker than VV.
    vh = np.where(water, -28.0, -15.0) + rng.normal(0, 1.5, (size, size))
    # Stack the two bands into one (2, H, W) image.
    img = np.stack([vv, vh]).astype(np.float32)
    # Set about 1 % of the pixels (in both bands) to NaN to test the cleaning code.
    img[:, rng.random((size, size)) < 0.01] = np.nan  # a few missing pixels, like real data
    # Label: 1 inside the disc, 0 outside.
    label = water.astype(np.int16)
    # Mark the first two rows as "no data" to test that -1 is ignored.
    label[:2] = -1  # a strip of "no data"
    return img, label


def write_tif(path: Path, arr: np.ndarray) -> None:
    """Save a numpy array as a GeoTIFF file (creating its folder if needed).

    Args:
        path: output file. arr: (bands, H, W) image or (H, W) label.
    """
    # rasterio wants (bands, H, W); a 2-D label gets a band axis added in front with [None].
    arr = arr if arr.ndim == 3 else arr[None]  # labels have 1 band
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open a new GeoTIFF for writing ("w") with the right size, band count and data type.
    # crs/transform give it a dummy geographic position (required by the format, not used by our code).
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype=arr.dtype, crs="EPSG:4326", transform=from_origin(0, 0, 1e-4, 1e-4)) as dst:
        dst.write(arr)


# split -> list of (event, number of chips). Bolivia only appears in its own split, like the real data.
FAKE_SPLITS = {
    "flood_train_data.csv": [("Alpha", 6), ("Beta", 6)],  # 12 training chips
    "flood_valid_data.csv": [("Alpha", 2), ("Gamma", 2)],  # 4 validation chips
    "flood_test_data.csv": [("Beta", 2), ("Gamma", 2)],  # 4 test chips
    "flood_bolivia_data.csv": [("Bolivia", 3)],  # 3 chips of the held-out event
}


# scope="session": the dataset is written only once and shared by all tests (faster).
@pytest.fixture(scope="session")
def fake_data(tmp_path_factory) -> Path:
    """Write the fake dataset once for the whole test session and return its root folder.

    tmp_path_factory is a built-in pytest fixture that creates temporary folders, deleted later by pytest.
    """
    # A fresh temporary folder named like the real dataset folder.
    root = tmp_path_factory.mktemp("sen1floods11")
    # Seeded generator -> the same fake dataset on every run.
    rng = np.random.default_rng(0)
    # Running number used to give every chip a unique id (Alpha_100001, Alpha_100002, ...).
    chip_number = 100000
    # One CSV file per split.
    for csv_name, events in FAKE_SPLITS.items():
        # Lines of this CSV file.
        rows = []
        # For each event in this split, create n chips.
        for event, n in events:
            for _ in range(n):
                chip_number += 1
                chip_id = f"{event}_{chip_number}"
                # Make the fake image + label and save them with the real file-name pattern.
                img, label = make_chip(rng)
                write_tif(root / "S1Hand" / f"{chip_id}_S1Hand.tif", img)
                write_tif(root / "LabelHand" / f"{chip_id}_LabelHand.tif", label)
                # Same row format as the real split CSVs: "image.tif,label.tif".
                rows.append(f"{chip_id}_S1Hand.tif,{chip_id}_LabelHand.tif")
        # Write the CSV (one row per line) into root/splits/.
        (root / "splits").mkdir(exist_ok=True)
        (root / "splits" / csv_name).write_text("\n".join(rows))
    return root


@pytest.fixture
def cfg(fake_data, tmp_path) -> dict:
    """The project config, pointed at the fake data and made small enough to run fast on a CPU.

    A new cfg (and a new empty output folder, tmp_path) is made for every test, so tests do not affect each other.
    """
    # Start from the real config.yaml so tests use the real setting names.
    c = load_config()
    # Read the fake data; write outputs into this test's own temporary folder (never the real outputs/).
    c["paths"] = {"data_dir": str(fake_data), "output_dir": str(tmp_path / "outputs")}
    # Random starting weights instead of downloading ImageNet weights.
    c["model"]["encoder_weights"] = None  # no download in tests
    # Tiny Random Forest: 10 small trees on 5000 pixels.
    c["baselines"].update(rf_trees=10, rf_max_depth=6, rf_pixels=5000)
    # Only 2 epochs, small batches, no mixed precision (CPU).
    c["train"].update(epochs=2, batch_size=4, amp=False)
    return c
