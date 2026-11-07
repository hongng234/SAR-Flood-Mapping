"""Tests for src/baselines.py (Otsu threshold and Random Forest).

The fake chips from tests/conftest.py have water at about -22 dB and land at about -8 dB, so a correct
threshold must fall between them and both baselines should find the water almost perfectly.
"""

import numpy as np  # arrays

from src.baselines import (
    fit_baselines,
    fit_global_otsu,
    load_baselines,
    pixel_features,
    predict_otsu_per_chip,
    predict_rf,
)
from src.data import clean_db, get_split, read_chip, read_label
from src.evaluate import iou  # IoU of a single chip
from tests.conftest import make_chip  # builds one fake chip + label


def test_otsu_finds_water_on_fake_chip():
    """On a clearly two-toned chip (dark lake, bright land), per-chip Otsu should find the lake."""
    # A larger fake chip (128 px) with a fixed seed.
    img, label = make_chip(np.random.default_rng(1), size=128)
    # Almost perfect overlap expected.
    assert iou(predict_otsu_per_chip(clean_db(img)), label) > 0.95


def test_otsu_constant_chip_predicts_no_water():
    """A chip where every pixel is -10 dB has no histogram to split: the code must return 'no water', not crash."""
    # np.full builds a (2, 8, 8) array filled with -10.0; .sum() == 0 means no pixel was called water.
    assert predict_otsu_per_chip(np.full((2, 8, 8), -10.0)).sum() == 0


def test_global_otsu_between_water_and_land(cfg):
    """The threshold learned from the train chips must lie between the water and land brightness."""
    assert -20 < fit_global_otsu(cfg) < -10  # fake water is -22 dB, land -8 dB


def test_pixel_features():
    """pixel_features returns 5 features per pixel, and feature 2 really is VV - VH."""
    img, _ = make_chip(np.random.default_rng(2))
    f = pixel_features(clean_db(img))
    # 64 x 64 pixels, 5 features each.
    assert f.shape == (64, 64, 5)
    # "..." means "all leading axes": compare feature 2 with feature 0 minus feature 1 for every pixel.
    np.testing.assert_allclose(f[..., 2], f[..., 0] - f[..., 1], rtol=1e-6)  # VV - VH


def test_fit_save_load_and_predict(cfg):
    """Full baseline round trip: fit on train -> save to disk -> load -> predict a test chip well."""
    # Writes otsu.json and rf.pkl into the test's temporary output folder.
    fit_baselines(cfg)
    threshold, rf = load_baselines(cfg)
    # First chip of the fake test split.
    s = get_split(cfg, "test")[0]
    img = clean_db(read_chip(s.image_path))
    # The loaded threshold is sensible, and the loaded Random Forest finds the lake.
    assert -20 < threshold < -10
    assert iou(predict_rf(rf, img), read_label(s.label_path)) > 0.8
