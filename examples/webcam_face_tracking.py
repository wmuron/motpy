import os
from typing import Sequence
from urllib.request import urlretrieve

import cv2
from motpy import Detection, MultiObjectTracker, NpImage, Box
from motpy.core import setup_logger
from motpy.detector import BaseObjectDetector
from motpy.testing_viz import draw_detection, draw_track

"""

    Example uses built-in camera (0) and baseline face detector from OpenCV (10 layer ResNet SSD)
    to present the library ability to track a face of the user

"""

logger = setup_logger(__name__, 'DEBUG', is_main=True)


WEIGHTS_URL = 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
WEIGHTS_PATH = 'opencv_face_detector.caffemodel'
CONFIG_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
CONFIG_PATH = 'deploy.prototxt'


class FaceDetector(BaseObjectDetector):
    def __init__(self,
                 weights_url: str = WEIGHTS_URL,
                 weights_path: str = WEIGHTS_PATH,
                 config_url: str = CONFIG_URL,
                 config_path: str = CONFIG_PATH,
                 conf_threshold: float = 0.5) -> None:
        super(FaceDetector, self).__init__()

        if not os.path.isfile(weights_path) or not os.path.isfile(config_path):
            logger.debug('downloading model...')
            urlretrieve(weights_url, weights_path)
            urlretrieve(config_url, config_path)

        self.net = cv2.dnn.readNetFromCaffe(config_path, weights_path)

        # specify detector hparams
        self.conf_threshold = conf_threshold

    def process_image(self, image: NpImage) -> Sequence[Detection]:
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        self.net.setInput(blob)
        detections = self.net.forward()

        # convert output from OpenCV detector to tracker expected format [xmin, ymin, xmax, ymax]
        out_detections = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                xmin = int(detections[0, 0, i, 3] * image.shape[1])
                ymin = int(detections[0, 0, i, 4] * image.shape[0])
                xmax = int(detections[0, 0, i, 5] * image.shape[1])
                ymax = int(detections[0, 0, i, 6] * image.shape[0])
                out_detections.append(Detection(box=[xmin, ymin, xmax, ymax], score=confidence))
            
        return out_detections


def run():
    # prepare multi object tracker
    model_spec = {'order_pos': 1, 'dim_pos': 2,
                  'order_size': 0, 'dim_size': 2,
                  'q_var_pos': 5000., 'r_var_pos': 0.1}

    dt = 1 / 15.0  # assume 15 fps
    tracker = MultiObjectTracker(dt=dt, model_spec=model_spec)

    # open camera
    cap = cv2.VideoCapture(0)

    face_detector = FaceDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)

        # run face detector on current frame
        detections = face_detector.process_image(frame)
        logger.debug(f'detections: {detections}')

        tracker.step(detections)
        tracks = tracker.active_tracks(min_steps_alive=3)
        logger.debug(f'tracks: {tracks}')

        # preview the boxes on frame
        for det in detections:
            draw_detection(frame, det)

        for track in tracks:
            draw_track(frame, track)

        cv2.imshow('frame', frame)

        # stop demo by pressing 'q'
        if cv2.waitKey(int(1000 * dt)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
