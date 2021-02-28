

import os
import random
import time

import cv2
import fire
import pandas as pd
from motpy import Box, Detection, MultiObjectTracker
from motpy.core import setup_logger
from motpy.testing_viz import draw_detection, draw_track

"""

    MOT16 tracking demo

    Usage:
        python examples/mot16_challange.py --dataset_root=~/Downloads/MOT16 --seq_id=11

    Note: this is just a demo, the script does not evaluate the tracking on MOT16 dataset.
    Also, since provided by MOT16 `predictions` do not represent (IMO) the current state
    of modern detectors, the demo utilizes ground truth + noise as input to the tracker;
    feel free to use `sel=det` to check the 'real' MOT16 predictions, but keep in mind that
    tracker is not optimized at all for such noisy predictions.

"""

logger = setup_logger(__name__, is_main=True)

COL_NAMES = ['frame_idx', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']

ALLOWED_SEQ_IDS = set(['02', '04', '05', '09', '10', '11', '13'])


def read_video_frame(directory, frame_idx):
    """ reads MOT16 formatted frame """
    fname = f'{frame_idx:06}.jpg'
    fpath = os.path.join(directory, fname)
    return cv2.imread(fpath)


def read_detections(path, drop_detection_prob: float = 0.0, add_detection_noise: float = 0.0):
    """ parses and converts MOT16 benchmark annotations to known [xmin, ymin, xmax, ymax] format """
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
        fps: float = 30.0,
        split: str = 'train',
        seq_id: str = '04',
        sel: str = 'gt',
        drop_detection_prob: float = 0.1,
        add_detection_noise: float = 5.0):
    """ parses detections, loads frames, runs tracking and visualizes the tracked objects """

    dataset_root = os.path.expanduser(dataset_root)
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
        model_spec='constant_acceleration_and_static_box_size_2d',
        matching_fn_kwargs={'min_iou': 0.25})

    # tracking loop
    while True:
        # read detections for a given frame
        try:
            frame_idx, detections = next(dets_gen)
        except Exception as e:
            logger.warning('finished reading the sequence')
            logger.debug(f'exception: {e}')
            break

        # read the frame for a given index
        frame = read_video_frame(frames_dir, frame_idx)
        if frame is None:
            continue

        # provide the MOT tracker with predicted detections
        t1 = get_miliseconds()
        active_tracks = tracker.step(detections)
        ms_elapsed = get_miliseconds() - t1
        logger.debug('step duration: %dms' % ms_elapsed)

        # visualize predictions and tracklets
        for det in detections:
            draw_detection(frame, det)

        for track in active_tracks:
            draw_track(frame, track)

        cv2.imshow('preview', frame)

        # stop execution on q
        key = cv2.waitKey(int(1000 / fps))
        if key == ord('q'):
            logger.info('early stopping')
            break


if __name__ == "__main__":
    fire.Fire(run)
