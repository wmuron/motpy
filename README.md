# motpy - simple multi object tracking library

Project is meant to provide a simple yet powerful baseline for multiple object tracking without the hassle of writing the obvious algorithm stack yourself.

![2D tracking preview](assets/mot16_challange.gif)

_video source: <https://motchallenge.net/data/MOT16/> - sequence 11_

## Features

    - tracking by detection paradigm
    - IOU + (optional) feature similarity matching strategy
    - Kalman filter used to model object trackers
    - each object is modeled as a center point (n-dimensional) and its size (n-dimensional); e.g. 2D position with width and height would be the most popular use case for bounding boxes tracking
    - seperately configurable system order for object position and size (currently 0th, 1st and 2nd order systems are allowed)
    - quite fast, more than realtime performance even on Raspberry Pi

## Installation

### Latest release

```bash
pip install motpy
```

#### Additional installation steps on Raspberry Pi

You might need to have to install following dependencies on RPi platform:

```bash
sudo apt-get install python-scipy
sudo apt install libatlas-base-dev
```

### Develop

```bash
git clone https://github.com/wmuron/motpy
cd motpy 
make install-develop # to install editable version of library
make test # to run all tests
```

## Example usage

### 2D tracking - synthetic example

Run demo example of tracking N objects in 2D space. In the ideal world it will show a bunch of colorful objects moving on a grey canvas in various directions, sometimes overlapping, sometimes not. Each object is detected from time to time (green box) and once it's being tracked by motpy, its track box is drawn in red with an ID above.

```bash
make demo
```

<https://user-images.githubusercontent.com/5874874/134305624-d6358cb1-39f8-4499-8a7b-64745f4795a6.mp4>

### Detect and track objects in the video

-   example uses a COCO-trained model provided by torchvision library
-   to run this example, you'll have to install `requirements_dev.txt` dependencies (`torch`, `torchvision`, etc.)
-   to run on CPU, specify `--device=cpu` 

```bash
python examples/detect_and_track_in_video.py \
            --video_path=./assets/video.mp4 \
            --detect_labels=['car','truck'] \
            --tracker_min_iou=0.15 \
            --device=cuda
```

<https://user-images.githubusercontent.com/5874874/134303165-b6835c8a-9cfe-486c-b79f-499f638c0a71.mp4>

_video source: <https://www.youtube.com/watch?v=PGMu_Z89Ao8/>, a great YT channel created by J Utah_

### MOT16 challange tracking

1.  Download MOT16 dataset from `https://motchallenge.net/data/MOT16/` and extract to `~/Downloads/MOT16` directory,
2.  Type the command: 
    ```bash
    python examples/mot16_challange.py --dataset_root=~/Downloads/MOT16 --seq_id=11
    ```
    This will run a simplified example where a tracker processes artificially corrupted ground-truth bounding boxes from sequence 11; you can preview the expected results in the beginning of the README file.

### Face tracking on webcam

Run the following command to start tracking your own face.

```bash
python examples/webcam_face_tracking.py
```

## Basic usage

A minimal tracking example can be found below:

```python
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

```

## Customization

To adapt the underlying motion model used to keep each object, you can pass a dictionary `model_spec` to `MultiObjectTracker`, which will be used to initialize each object tracker at its creation time. The exact parameters can be found in definition of `motpy.model.Model` class. 
See the example below, where I've adapted the motion model to better fit the typical motion of face in the laptop camera and decent face detector.

```python
model_spec = {
        'order_pos': 1, 'dim_pos': 2, # position is a center in 2D space; under constant velocity model
        'order_size': 0, 'dim_size': 2, # bounding box is 2 dimensional; under constant velocity model
        'q_var_pos': 1000., # process noise
        'r_var_pos': 0.1 # measurement noise
    }

tracker = MultiObjectTracker(dt=0.1, model_spec=model_spec)
```

The simplification used here is that the object position and size can be treated and modeled independently; hence you can use even 2D bounding boxes in 3D space.

Feel free to tune the parameter of Q and R matrix builders to better fit your use case.

## Tested platforms

    - Linux (Ubuntu)
    - macOS (Catalina)
    - Raspberry Pi (4)

## Things to do

    - [x] Initial version
    - [ ] Documentation
    - [ ] Performance optimization
    - [x] Multiple object classes support via instance-level class_id counting
    - [x] Allow tracking without Kalman filter
    - [x] Easy to use and configurable example of video processing with off-the-shelf object detector

## References, papers, ideas and acknowledgements

    - https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/
    - http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf
    - https://arxiv.org/abs/1602.00763
