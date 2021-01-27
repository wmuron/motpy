
import numpy as np
import scipy
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from motpy.core import Detection, setup_logger
from motpy.model import Model
from motpy.tracker import MultiObjectTracker, match_by_cost_matrix
from scipy.linalg import block_diag

from utils import assert_almost_equal

logger = setup_logger(__name__)


def test_builders():
    """ model 1 """
    m1 = Model(0.1, 1, 2, 0, 2, r_var_pos=0.1, r_var_size=0.3, p_cov_p0=100.)

    assert m1.state_length == 6
    assert m1.measurement_length == 4

    F1 = m1.build_F()
    F1_exp = np.array([[1., 0.1, 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 0.],
                       [0., 0., 1., 0.1, 0., 0.],
                       [0., 0., 0., 1., 0., 0.],
                       [0., 0., 0., 0., 1., 0.],
                       [0., 0., 0., 0., 0., 1.]])
    assert_almost_equal(F1_exp, F1)

    H1 = m1.build_H()
    H1_exp = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1]])
    assert_almost_equal(H1_exp, H1)

    _ = m1.build_Q()

    R1 = m1.build_R()
    R1_exp = np.array([[0.1, 0, 0, 0],
                       [0, 0.1, 0, 0.],
                       [0, 0, 0.3, 0.],
                       [0, 0, 0, 0.3]])
    assert_almost_equal(R1, R1_exp)

    _ = m1.build_P()

    """ model 2 """
    m2 = Model(0.1, 2, 1, 1, 1)
    F2 = m2.build_F()
    F2_exp = np.array([[1., 0.1, 0.005, 0., 0.],
                       [0., 1., 0.1, 0., 0.],
                       [0., 0., 1., 0., 0.],
                       [0., 0., 0., 1., 0.1],
                       [0., 0., 0., 0., 1.]])
    assert_almost_equal(F2_exp, F2)


def test_state_to_observation_converters():
    # 2d boxes
    model = Model(0.1, 1, 2, 0, 2)

    # xmin ymin xmax ymax
    box = [10, 10, 20, 30]
    x = model.box_to_x(box)
    assert_almost_equal(np.array([15., 0., 20., 0., 10., 20.]), x)
    box_ret = model.x_to_box(x)
    assert_almost_equal(box_ret, box)

    # 3d boxes
    model = Model(0.1, 1, 3, 0, 3)

    # xmin ymin zmin xmax ymax zmax
    box = [10, 10, 10, 20, 30, 40]
    x = model.box_to_x(box)
    assert_almost_equal(np.array([15., 0., 20., 0., 25., 0., 10., 20., 30.]), x)
    box_ret = model.x_to_box(x)
    assert_almost_equal(box_ret, box)

    # 2d position, 3d boxes (z dimension is only about length)
    model = Model(0.1, 1, 2, 0, 3)
    box = [10, 10, -25, 20, 30, 25]
    x = model.box_to_x(box)
    x_exp = np.array([15., 0., 20., 0., 10., 20., 50.])
    assert_almost_equal(x_exp, x)
    box_ret = model.x_to_box(x)
    assert_almost_equal(box_ret, box)


def test_box_to_z():
    model = Model(0.1, 1, 2, 0, 2)

    box = [10, 10, 20, 20]
    z_exp = [15, 15, 10, 10]
    assert_almost_equal(model.box_to_z(box=box), z_exp)

    # size not matching dimensions
    box = [10, 10, 10, 20, 20, 20]
    try:
        # TODO cleanup
        model.box_to_z(box=box)
        raise
    except AssertionError:
        pass
    except Exception:
        raise

    # 3d position, 2d size
    model = Model(0.1, 1, 3, 1, 2)
    box = [10, 10, 0, 20, 20, 50]
    z = model.box_to_z(box)
    z_exp = [15, 15, 25, 10, 10]
    assert_almost_equal(z, z_exp)
