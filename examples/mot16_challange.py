

import os
import random
import time

import cv2
import fire
import pandas as pd
from loguru import logger

from motpy import Box, Detection, MultiObjectTracker
from motpy.testing_viz import draw_detection, draw_track

"""

    MOT16 tracking demo

    Usage:
        python examples/mot16_challange.py --dataset_root ~/Downloads/MOT16 --seq_id 11

    Note: this is just a demo, the script does not evaluate the tracking on MOT16 dataset.

"""


COL_NAMES = ['frame_idx', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']

ALLOWED_SEQ_IDS = set(['02', '04', '05', '09', '10', '11', '13'])


def read_video_frame(directory, frame_idx):
    """ reads MOT16 formatted frame """
    fname = f'{frame_idx:06}.jpg'
    fpath = os.path.join(directory, fname)
    return cv2.imread(fpath)


def read_detections(path, drop_detection_prob: float = 0.0, add_detection_noise: float = 0.0):
    path = os.path.expanduser(path)
    logger.debug('reading detections from %s' % path)
    if not os.path.isfile(path):
        raise ValueError('file does not exist')

    df = pd.read_csv(path, names=COL_NAMES)

    max_frame = df.frame_idx.max()
    for frame_idx in range(max_frame):
        detections = []
        for _, row in df[df.frame_idx == frame_idx].iterrows():
            if random.random() < drop_detection_prob:
                continue

            box = [row.bb_left, row.bb_top,
                   row.bb_left + row.bb_width,
                   row.bb_top + row.bb_height]

            if add_detection_noise > 0:
                for i in range(4):
                    box[i] += random.uniform(-add_detection_noise, add_detection_noise)

            detections.append(Detection(box=box))

        yield frame_idx, detections


def get_miliseconds():
    return int(round(time.time() * 1000))


def run(
        dataset_root: str,
        fps=30,
        split: str = 'train',
        seq_id='04',
        sel='gt',
        drop_detection_prob: float = 0.1,
        add_detection_noise: float = 5.0):

    if not os.path.isdir(dataset_root):
        logger.error('%s does not exist' % dataset_root)
        exit(-1)

    if str(seq_id) not in ALLOWED_SEQ_IDS:
        logger.error('unknown MOT16 sequence: %s' % str(seq_id))
        exit(-1)

    dataset_root2 = f'{dataset_root}/{split}/MOT16-{seq_id}'
    frames_dir = f'{dataset_root2}/img1'
    logger.info(f'reading video frames from {frames_dir}')

    dets_path = f'{dataset_root2}/{sel}/{sel}.txt'
    dets_gen = read_detections(
        dets_path,
        drop_detection_prob=drop_detection_prob,
        add_detection_noise=add_detection_noise)

    tracker = MultiObjectTracker(
        dt=1 / fps, tracker_kwargs={'max_staleness': 15},
        model_spec='2d_constant_acceleration+static_box_size')

    # TODO cleanup
    tracker.matching_fn.min_iou = 0.25

    # tracking loop
    while True:
        frame_idx, detections = next(dets_gen)
        frame = read_video_frame(frames_dir, frame_idx)
        if frame is None:
            continue

        t1 = get_miliseconds()
        active_tracks = tracker.step(detections)
        ms_elapsed = get_miliseconds() - t1
        logger.debug('step duration: %dms' % ms_elapsed)

        for det in detections:
            draw_detection(frame, det)

        for track in active_tracks:
            draw_track(frame, track)

        cv2.imshow('preview', frame)
        key = cv2.waitKey(int(1000 / fps))
        if key == ord('q'):
            break


if __name__ == "__main__":
    fire.Fire(run)
