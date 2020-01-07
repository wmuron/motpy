import collections
import sys
from typing import Optional

import numpy as np
from loguru import logger

""" types """

# Box is of shape (1,2xdim), e.g. for dim=2 [xmin, ymin, xmax, ymax] format is accepted
Box = np.ndarray

# Vector is of shape (1, N)
Vector = np.ndarray

# Track is meant as an output from the object tracker
Track = collections.namedtuple('Track', 'id box')


class Detection:
    # Detection is to be an input the the tracker
    def __init__(
            self,
            box: Box,
            score: Optional[float] = None,
            feature: Optional[Vector] = None):
        self.box = box
        self.score = score
        self.feature = feature

    def __repr__(self):
        fmt = "(box) %s\n(score) %.2f\n(feature) %s"
        return fmt % (str(self.box), self.score, str(self.feature))


""" utils """


def set_log_level(level: str) -> None:
    logger.remove()
    logger.add(sys.stdout, level=level)
