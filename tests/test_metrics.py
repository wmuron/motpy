
import numpy as np
from motpy.metrics import angular_similarity, calculate_iou

from utils import assert_almost_equal, current_milli_time


def test_ious():
    b1 = [[10, 20]]
    b2 = [[10, 21], [30, 40]]
    iou_1d = calculate_iou(b1, b2, dim=1)
    assert_almost_equal(iou_1d, np.array([[0.9091, 0.]]))

    b1 = [[20.1, 20.1, 30.1, 30.1], [15, 15, 25, 25]]
    b2 = [[10, 10, 20, 20]]
    iou_2d = calculate_iou(b1, b2, dim=2)
    assert_almost_equal(iou_2d, [[0], [0.1429]])

    b1 = [[10, 10, 10, 20, 20, 20]]
    b2 = [[10, 11, 10.2, 21, 19.9, 20.3],
          [30, 30, 30, 90, 90, 90]]
    iou_3d = calculate_iou(b1, b2, dim=3)
    assert_almost_equal(iou_3d, [[0.7811, 0]])


def test_ious_large():
    # benchmarking test
    t = current_milli_time()
    for _ in range(100):
        b1 = np.random.randn(30, 4)
        b2 = np.random.randn(20, 4)
        iou = calculate_iou(b1, b2, dim=2)
        assert iou.shape == (30, 20)

    tdiff = current_milli_time() - t
    assert tdiff < 100


def test_angular_similarity():
    a = [[10, 10], [0, 100]]
    b = [[0, 10], [-10, -10]]

    mat = angular_similarity(a, b)
    mat_exp = np.array([[0.8536, 0.],
                        [1., 0.1464]])

    assert_almost_equal(mat, mat_exp)
