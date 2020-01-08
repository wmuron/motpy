import numpy as np

import motpy
from motpy import MultiObjectTracker, Detection

# format [xmin, ymin, xmax, ymax]
object_box = np.array([1, 1, 10, 10])

tracker = MultiObjectTracker(dt=0.1)

for step in range(10):
    # let's simulate object movement
    object_box += 1

    tracker.step(detections=[Detection(box=object_box)])
    tracks = tracker.active_tracks()

    print('MOT tracker tracks %d objects' % len(tracks))
    print('first track box: %s' % str(tracks[0].box))
