import functools
import numpy as np
import time


assert_almost_equal = functools.partial(
    np.testing.assert_almost_equal, decimal=4)


def current_milli_time():
    return round(time.time() * 1000)
