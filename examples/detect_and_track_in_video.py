import os
import time
from typing import Sequence

import cv2
import fire
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from motpy import Detection, ModelPreset, MultiObjectTracker, NpImage
from motpy.core import setup_logger
from motpy.detector import BaseObjectDetector
from motpy.testing_viz import draw_detection, draw_track
from motpy.utils import ensure_packages_installed

from coco_labels import get_class_ids

ensure_packages_installed(['torch', 'torchvision', 'cv2'])


"""

    Usage:

        python examples/detect_and_track_in_video.py \
            --video_path=./assets/video.mp4 \
            --detect_labels=['car','truck'] \
            --tracker_min_iou=0.2 \
            --architecture=fasterrcnn \
            --device=cuda

"""


logger = setup_logger(__name__, 'DEBUG', is_main=True)


class CocoObjectDetector(BaseObjectDetector):
    """ A wrapper of torchvision example object detector trained on COCO dataset """

    def __init__(self,
                 class_ids: Sequence[int],
                 confidence_threshold: float = 0.5,
                 architecture: str = 'ssdlite320',
                 device: str = 'cpu'):

        self.confidence_threshold = confidence_threshold
        self.device = device
        self.class_ids = class_ids
        assert len(self.class_ids) > 0, f'select more than one class_ids'

        if architecture == 'ssdlite320':
            self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
        elif architecture == 'fasterrcnn':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        else:
            raise NotImplementedError(f'unknown architecture: {architecture}')

        self.model = self.model.eval().to(device)

        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _predict(self, image):
        image = self.input_transform(image).to(self.device).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(image)

        outs = [pred[0][attr].detach().cpu().numpy() for attr in ['boxes', 'scores', 'labels']]

        sel = np.logical_and(
            np.isin(outs[2], self.class_ids),  # only selected class_ids
            outs[1] >= self.confidence_threshold)  # above confidence threshold

        return [outs[idx][sel].astype(to_type) for idx, to_type in enumerate([float, int, float])]

    def process_image(self, image: NpImage) -> Sequence[Detection]:
        t0 = time.time()
        boxes, scores, class_ids = self._predict(image)
        elapsed = (time.time() - t0) * 1000.
        logger.debug(f'inference time: {elapsed:.3f} ms')
        return [Detection(box=b, score=s, class_id=l) for b, s, l in zip(boxes, scores, class_ids)]


def read_video_file(video_path: str):
    video_path = os.path.expanduser(video_path)
    cap = cv2.VideoCapture(video_path)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    return cap, video_fps


def run(video_path: str, detect_labels,
        video_downscale: float = 1.,
        architecture: str = 'ssdlite320',
        confidence_threshold: float = 0.5,
        tracker_min_iou: float = 0.25,
        show_detections: bool = False,
        track_text_verbose: int = 0,
        device: str = 'cpu',
        viz_wait_ms: int = 1):
    # setup detector, video reader and object tracker
    detector = CocoObjectDetector(class_ids=get_class_ids(detect_labels), confidence_threshold=confidence_threshold, architecture=architecture, device=device)
    cap, cap_fps = read_video_file(video_path)
    tracker = MultiObjectTracker(
        dt=1 / cap_fps,
        tracker_kwargs={'max_staleness': 5},
        model_spec={'order_pos': 1, 'dim_pos': 2,
                    'order_size': 0, 'dim_size': 2,
                    'q_var_pos': 5000., 'r_var_pos': 0.1},
        matching_fn_kwargs={'min_iou': tracker_min_iou,
                            'multi_match_min_iou': 0.93})

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, fx=video_downscale, fy=video_downscale, dsize=None, interpolation=cv2.INTER_AREA)

        # detect objects in the frame
        detections = detector.process_image(frame)

        # track detected objects
        _ = tracker.step(detections=detections)
        active_tracks = tracker.active_tracks(min_steps_alive=3)

        # visualize and show detections and tracks
        if show_detections:
            for det in detections:
                draw_detection(frame, det)

        for track in active_tracks:
            draw_track(frame, track, thickness=2, text_at_bottom=True, text_verbose=track_text_verbose)

        cv2.imshow('frame', frame)
        c = cv2.waitKey(viz_wait_ms)
        if c == ord('q'):
            break


if __name__ == '__main__':
    fire.Fire(run)
