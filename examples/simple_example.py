import numpy as np

from motpy import Detection, MultiObjectTracker

# create a simple bounding box with format of [xmin, ymin, xmax, ymax]
object_box = np.array([1, 1, 10, 10])

# create a multi object tracker with a specified step time of 100ms
tracker = MultiObjectTracker(dt=0.1)

for step in range(10):
    # let's simulate object movement by 1 unit (e.g. pixel)
    object_box += 1

    # update the state of the multi-object-tracker tracker
    # with the list of bounding boxes
    tracker.step(detections=[Detection(box=object_box)])

    # retrieve the active tracks from the tracker (you can customize
    # the hyperparameters of tracks filtering by passing extra arguments)
    tracks = tracker.active_tracks()

    print('MOT tracker tracks %d objects' % len(tracks))
    print('first track box: %s' % str(tracks[0].box))
