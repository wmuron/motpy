import functools
import numpy as np

assert_almost_equal = functools.partial(
    np.testing.assert_almost_equal, decimal=4)
