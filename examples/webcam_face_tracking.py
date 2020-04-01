import os
import urllib
from urllib.request import urlretrieve

import cv2
from loguru import logger

from motpy import Detection, MultiObjectTracker
from motpy.testing_viz import draw_detection, draw_track


"""

    Example uses built-in camera (0) and baseline face detector from OpenCV (Haar Cascade) to present
    the library ability to track a face of the user

"""


FACE_DETECTOR_HAAR_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
FACE_DETECTOR_HAAR_PATH = 'haarcascade_frontalface_default.xml'


def run():
    # prepare multi object tracker
    model_spec = {'order_pos': 1, 'dim_pos': 2,
                  'order_size': 0, 'dim_size': 2,
                  'q_var_pos': 5000., 'r_var_pos': 0.1}

    dt = 1 / 24.0
    tracker = MultiObjectTracker(dt=dt, model_spec=model_spec)

    # prepare baseline face detector
    if not os.path.isfile(FACE_DETECTOR_HAAR_PATH):
        logger.info(
            f'downloading detector from {FACE_DETECTOR_HAAR_URL} to {FACE_DETECTOR_HAAR_PATH}')
        urlretrieve(FACE_DETECTOR_HAAR_URL, FACE_DETECTOR_HAAR_PATH)

    face_cascade = cv2.CascadeClassifier(FACE_DETECTOR_HAAR_PATH)

    # open camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)

        # detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # convert output from OpenCV detector to tracker expected format [xmin, ymin, xmax, ymax]
        detections = [Detection(box=[x, y, x + w, y + h]) for (x, y, w, h) in faces]
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
        if cv2.waitKey(int(1000 * dt)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
