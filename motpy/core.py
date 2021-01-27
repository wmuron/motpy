import collections
import logging
import os
import sys
from typing import Optional

import numpy as np

""" types """

# Box is of shape (1,2xdim), e.g. for dim=2 [xmin, ymin, xmax, ymax] format is accepted
Box = np.ndarray

# Vector is of shape (1, N)
Vector = np.ndarray

# Track is meant as an output from the object tracker
Track = collections.namedtuple('Track', 'id box')

# numpy/opencv image alias
NpImage = np.ndarray


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
        fmt = "(detection: box=%s, score=%s, feature=%s)"
        return fmt % (str(self.box),
                      str(self.score) or 'none',
                      str(self.feature) or 'none')


""" utils """

LOG_FORMAT = "%(asctime)s\t%(threadName)s-%(name)s:%(levelname)s:%(message)s"


def setup_logger(name: str,
                 level: Optional[str] = None,
                 is_main: bool = False,
                 envvar_name: str = 'MOTPY_LOG_LEVEL'):
    if level is None:
        level = os.getenv(envvar_name)
        if level is None:
            print(f'[{name}] fallback to INFO log_level; set {envvar_name} envvar to override')
            level = 'INFO'
        else:
            print(f'[{name}] envvar {envvar_name} sets log level to {level}')

    level_val = logging.getLevelName(level)

    if is_main:
        logging.basicConfig(stream=sys.stdout, level=level_val, format=LOG_FORMAT)

    logger = logging.getLogger(name)
    logger.setLevel(level_val)

    return logger
