from typing import Sequence

import numpy as np
from motpy.tracker import Detection

Image = np.ndarray


class BaseObjectDetector:
    def __init__(self) -> None:
        pass

    def process_image(self, image: Image) -> Sequence[Detection]:
        raise NotImplementedError('subclass the BaseObjectDetector with your custom detector')
