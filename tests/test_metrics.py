import numpy as np
import pytest

from zgq.search import compute_recall as search_recall
from zgq.utils.metrics import compute_recall


def test_recall_perfect_and_zero():
    gt = np.array([[0, 1, 2], [3, 4, 5]])
    assert compute_recall(gt, gt, k=3) == pytest.approx(100.0)
    assert compute_recall(gt + 100, gt, k=3) == pytest.approx(0.0)


def test_recall_partial():
    gt = np.array([[0, 1, 2, 3]])
    pred = np.array([[0, 1, 9, 8]])
    assert compute_recall(pred, gt, k=4) == pytest.approx(50.0)
    assert search_recall(pred, gt, k=4) == pytest.approx(50.0)
