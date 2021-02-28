from collections import Counter

import numpy as np
import pytest
from motpy.core import Detection, setup_logger
from motpy.testing import data_generator
from motpy.tracker import (BasicMatchingFunction, MultiObjectTracker,
                           exponential_moving_average_fn, match_by_cost_matrix)
from numpy.testing import assert_array_equal

logger = setup_logger(__name__)


@pytest.mark.parametrize("num_objects", [2, 5])
@pytest.mark.parametrize("order_pos", [0, 1, 2])
@pytest.mark.parametrize("feature_similarity_beta", [None, 0.5])
def test_simple_tracking_objects(
        num_objects: int,
        order_pos: int,
        feature_similarity_beta: float,
        fps: int = 24,
        num_steps: int = 240):

    gen = data_generator(
        num_steps=num_steps,
        miss_prob=0.2,
        det_err_sigma=3.0,
        num_objects=num_objects,
        max_omega=0.01)

    dt = 1 / fps
    num_steps_warmup = 1.0 * fps  # 1 second of warmup
    model_spec = {'order_pos': order_pos, 'dim_pos': 2, 'order_size': 0, 'dim_size': 2}

    matching_fn = BasicMatchingFunction(feature_similarity_beta=feature_similarity_beta)
    tracker = MultiObjectTracker(
        dt=dt,
        model_spec=model_spec,
        matching_fn=matching_fn)
    history = {idx: [] for idx in range(num_objects)}
    for i in range(num_steps):
        dets_gt, dets_pred = next(gen)
        detections = [d for d in dets_pred if d.box is not None]
        active_tracks = tracker.step(detections=detections)

        if i <= num_steps_warmup:
            continue

        matches = match_by_cost_matrix(dets_gt, active_tracks)
        for m in matches:
            gidx, tidx = m[0], m[1]
            track_id = active_tracks[tidx].id
            history[gidx].append(track_id)

        assert len(tracker.trackers) == num_objects

    count_frames = (num_steps - num_steps_warmup)
    for gid in range(num_objects):
        c = Counter(history[gid])
        count_tracked = c.most_common(1)[0][1]
        logger.info('object %d mostly tracked in %.1f%% frames' %
                    (gid, (100.0 * count_tracked / count_frames)))
        assert count_tracked > 0.95 * count_frames


def test_tracker_diverges():
    box = np.array([0, 0, 10, 10])

    tracker = MultiObjectTracker(dt=0.1)
    tracker.step([Detection(box=box)])
    assert len(tracker.trackers) == 1
    first_track_id = tracker.active_tracks()[0].id

    # check valid tracker
    assert tracker.trackers[0].is_invalid == False
    tracker.trackers[0]._tracker.x[2] = np.nan
    assert tracker.trackers[0].is_invalid

    tracker.cleanup_trackers()
    assert len(tracker.trackers) == 0

    # pass invalid box
    tracker.step([Detection(box=box)])
    assert len(tracker.trackers) == 1
    assert tracker.active_tracks()[0].id != first_track_id


def test_exponential_moving_average():
    update_fn = exponential_moving_average_fn(0.5)

    # scalars
    assert update_fn(None, 100.) == 100.
    assert update_fn(50., None) == 50.
    assert update_fn(50., 100.) == 75.

    # sequences
    assert_array_equal(update_fn(None, [100., 50]), [100., 50])
    assert_array_equal(update_fn([80., 50], None), [80., 50])
    assert_array_equal(update_fn(np.array([80., 100]), [90, 90]), [85., 95])
