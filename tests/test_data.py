"""Tests for src/data.py, plus a check of the real split files when the data is downloaded.

Every function whose name starts with "test_" is run by pytest. "assert X" fails the test if X is False.
`cfg` in a test's parameters is the fixture from tests/conftest.py (config pointed at the fake dataset).
"""

import itertools  # combinations() gives every pair of splits
from pathlib import Path  # file paths

import numpy as np  # arrays
import pytest  # test framework (approx, skip, parametrize)

from main import load_config  # the real config, for the real-data test
from src.data import FloodDataset, augment, clean_db, compute_norm_stats, get_split, normalize, read_chip, read_label
from tests.conftest import SIZE  # side length of the fake chips (64)


def test_read_split_and_files(cfg):
    """get_split reads the CSV correctly, and read_chip / read_label return the expected shapes and values."""
    train = get_split(cfg, "train")
    # The fake train split has 6 Alpha + 6 Beta chips, Alpha first; .event takes the name before "_".
    assert len(train) == 12 and train[0].event == "Alpha"
    img, label = read_chip(train[0].image_path), read_label(train[0].label_path)
    # Image has 2 bands (VV, VH); label is a single 2-D map.
    assert img.shape == (2, SIZE, SIZE) and label.shape == (SIZE, SIZE)
    # The label contains only the allowed values ("<=" on sets means "is a subset of").
    assert set(np.unique(label)) <= {-1, 0, 1}


def test_clean_db_clips_and_fills_nan():
    """clean_db clips to the range, replaces NaN with the band median, and does not change its input."""
    # A tiny hand-made (2, 2, 2) image: band 0 has a NaN, a too-low (-80) and a too-high (5) value.
    img = np.array([[[np.nan, -80.0], [5.0, -10.0]], [[-20.0, -21.0], [-22.0, -23.0]]], dtype=np.float32)
    out = clean_db(img, (-50.0, 1.0))
    # No NaN left and everything inside [-50, 1].
    assert np.isfinite(out).all() and out.min() >= -50 and out.max() <= 1
    # After clipping band 0 holds -50, 1, -10; their median is -10, so the NaN becomes -10.
    assert out[0, 0, 0] == pytest.approx(-10.0)  # NaN -> median of (-50, 1, -10)
    # The original array still has its NaN: clean_db worked on a copy.
    assert np.isnan(img[0, 0, 0])  # input not modified


def test_normalize_gives_zero_mean_unit_std(cfg):
    """After normalising with the train statistics, the train data has mean ~0 and std ~1."""
    train = get_split(cfg, "train")
    stats = compute_norm_stats(train)
    # Normalise every train chip and stack them into one array (N, 2, H, W).
    x = np.stack([normalize(clean_db(read_chip(s.image_path)), stats) for s in train])
    # Mean close to 0 overall, and the VV band's std close to 1 (0.05 tolerance).
    assert abs(x.mean()) < 0.05 and abs(x[:, 0].std() - 1) < 0.05


# parametrize runs this test 5 times, with seed = 0, 1, 2, 3, 4, to try different random flips/rotations.
@pytest.mark.parametrize("seed", range(5))
def test_augment_keeps_image_and_mask_aligned(seed):
    """augment must transform image and mask in exactly the same way, otherwise labels no longer match pixels."""
    rng = np.random.default_rng(seed)
    # Random 8x8 mask with values -1, 0, 1.
    mask = rng.integers(-1, 2, (8, 8))
    # Build the image FROM the mask (band 0 = mask, band 1 = 2 * mask), so any misalignment is visible.
    img = np.stack([mask, mask * 2]).astype(np.float32)  # image built from the mask
    a_img, a_mask = augment(img, mask, rng)
    # After augmentation the relationship must still hold pixel by pixel.
    np.testing.assert_array_equal(a_img[0], a_mask)  # still identical after flips/rotations
    np.testing.assert_array_equal(a_img[1], a_mask * 2)


def test_dataset_item(cfg):
    """FloodDataset returns tensors of the right shape, without NaN, together with the chip id."""
    samples = get_split(cfg, "train")
    # [0] calls __getitem__(0): the first item.
    img, mask, chip_id = FloodDataset(samples, compute_norm_stats(samples), train=True)[0]
    assert tuple(img.shape) == (2, SIZE, SIZE) and tuple(mask.shape) == (SIZE, SIZE)
    # NaN were filled by clean_db; the first train chip is an Alpha chip.
    assert not img.isnan().any() and chip_id.startswith("Alpha_")


def test_real_splits_do_not_overlap():
    """No chip in two splits, and Bolivia never in train/val. Skipped if the data is not downloaded.

    This uses the REAL dataset (not the fake one) to confirm the official splits are clean.
    """
    cfg = load_config()
    # Skip (not fail) on a machine where `python main.py download` has not been run.
    if not (Path(cfg["paths"]["data_dir"]) / "splits").exists():
        pytest.skip("dataset not downloaded")
    # split name -> set of chip ids in that split.
    ids = {split: {s.chip_id for s in get_split(cfg, split)} for split in cfg["data"]["splits"]}
    # Check every pair of splits: their intersection ("&") must be empty.
    for a, b in itertools.combinations(ids, 2):
        assert not ids[a] & ids[b], f"{a} and {b} share chips"
    # No Bolivia chip among the chips used for training or model selection ("|" = union of sets).
    assert not any(c.startswith("Bolivia") for c in ids["train"] | ids["val"])
    # The official split sizes.
    assert [len(ids[s]) for s in ("train", "val", "test", "bolivia")] == [252, 89, 90, 15]
