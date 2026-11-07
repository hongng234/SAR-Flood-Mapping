"""Tests for the metrics in src/evaluate.py, checked against hand-computed values.

Metrics must be right, because every result in the README is computed with them.
Reminder: TP = water found, FP = false alarm, FN = missed water, TN = correct not-water; IoU = TP / (TP + FP + FN).
"""

import numpy as np  # arrays
import pytest  # pytest.approx compares floats with a small tolerance

from src.evaluate import Metrics, confusion, iou, scores


def test_confusion_counts():
    """Each of the 4 pixels falls in a different box, so each count must be 1."""
    pred = np.array([1, 1, 0, 0])
    label = np.array([1, 0, 1, 0])
    # Pixel by pixel: (1,1)=TP, (1,0)=FP, (0,1)=FN, (0,0)=TN.
    assert confusion(pred, label).tolist() == [1, 1, 1, 1]  # TP, FP, FN, TN


def test_scores_by_hand():
    """Compare every metric with the formula worked out by hand."""
    s = scores(np.array([6, 2, 4, 88]))  # TP=6, FP=2, FN=4
    assert s["iou"] == pytest.approx(6 / 12)  # 6 / (6 + 2 + 4)
    assert s["f1"] == pytest.approx(12 / 18)  # 2*6 / (2*6 + 2 + 4)
    assert s["precision"] == pytest.approx(6 / 8)  # 6 / (6 + 2)
    assert s["recall"] == pytest.approx(6 / 10)  # 6 / (6 + 4)


def test_no_data_pixels_are_ignored():
    """Pixels labelled -1 must not change the score, whatever is predicted there."""
    label = np.array([1, 0, -1, -1])
    pred = np.array([1, 0, 1, 0])  # predictions on the -1 pixels must not count
    # The two labelled pixels are both correct -> perfect IoU.
    assert iou(pred, label) == 1.0


def test_edge_cases():
    """Special cases: no water anywhere, all water missed, only false alarms."""
    # np.zeros(5, int) = five 0s, np.ones(5, int) = five 1s.
    zeros, ones = np.zeros(5, int), np.ones(5, int)
    assert iou(zeros, zeros) == 1.0  # no water anywhere: perfect agreement
    assert iou(zeros, ones) == 0.0  # missed all water
    assert iou(ones, zeros) == 0.0  # only false alarms


def test_micro_average_and_per_event():
    """Metrics sums the pixel counts over chips (micro average) and also keeps them per event."""
    m = Metrics()
    # A big chip predicted perfectly, and a small chip where all water is missed.
    m.update(np.ones(100), np.ones(100), "Big_1")  # 100 TP
    m.update(np.zeros(2), np.ones(2), "Small_2")  # 2 FN
    # Micro average = 100 / (100 + 2), not the mean of the chip IoUs (1.0 and 0.0) which would be 0.5.
    assert m.scores()["iou"] == pytest.approx(100 / 102)  # micro, not (1.0 + 0.0) / 2
    # The "Small" event on its own scores 0.
    assert scores(m.per_event["Small"])["iou"] == 0.0
