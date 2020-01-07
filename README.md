# motpy - simple multi object tracking library

Project is meant to provide a simple yet powerful baseline for multiple object tracking without the hassle of writing the obvious algorithm stack yourself.

## Features:

- tracking by detection paradigm
- IOU + (optional) feature similarity matching strategy
- Kalman filter used to model object trackers
- each object is modeled as a center point (n-dimensional) and its size (n-dimensional); e.g. 2D position with width and height would be the most popular use case for bounding boxes tracking
- seperately configurable system order for object position and size (currently 0th, 1st and 2nd order systems are allowed)
- quite fast, more than realtime performance even on Raspberry Pi

## Installation:

```bash
pip install motpy
```

### Develop:
```bash
git clone https://github.com/wmuron/motpy
cd motpy 
make install-develop # to install editable version of library
make test # to run all tests
```
## Basic usage

Run demo example of tracking N objects in 2D space. In the ideal world it will show a bunch of colorful objects moving on a grey canvas in various directions, sometimes overlapping, sometimes not. Each object is detected from time to time (green box) and once it's being tracked by motpy, its track box is drawn in red with an ID above.
```
make demo
```

Minimal tracking example below:

```python
import numpy as np

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

```

## Tested platforms
- Linux (Ubuntu)
- Raspberry Pi (4)

## Things to do:
- [x] Initial version
- [ ] Documentation
- [ ] Performance optimization
- [ ] Multiple object classes support

## References, papers, ideas and acknowledgements
- https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/
- http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf
- https://arxiv.org/abs/1602.00763