import uuid
from collections.abc import Iterable
from typing import Any, Optional, Sequence, Union

import numpy as np
import scipy
from filterpy.kalman import KalmanFilter

from motpy.core import Box, Detection, Track, Vector, setup_logger
from motpy.metrics import angular_similarity, calculate_iou
from motpy.model import Model, ModelPreset

logger = setup_logger(__name__)


def get_single_object_tracker(model: Model, x0: Optional[Vector] = None) -> KalmanFilter:
    """ returns Kalman-based tracker based on a specified motion model spec.
        e.g. for spec = {'order_pos': 1, 'dim_pos': 2, 'order_size': 0, 'dim_size': 1}
        we expect the following setup:
        state x, x', y, y', w, h
        where x and y are centers of boxes
              w and h are width and height
    """

    tracker = KalmanFilter(dim_x=model.state_length,
                           dim_z=model.measurement_length)
    tracker.F = model.build_F()
    tracker.Q = model.build_Q()
    tracker.H = model.build_H()
    tracker.R = model.build_R()
    tracker.P = model.build_P()

    if x0 is not None:
        tracker.x = x0

    return tracker


DEFAULT_MODEL_SPEC = ModelPreset.constant_velocity_and_static_box_size_2d.value


def exponential_moving_average_fn(gamma: float) -> Any:
    def fn(old, new):
        if new is None:
            return old

        if isinstance(new, Iterable):
            new = np.array(new)

        if old is None:
            return new  # first call
        else:
            return gamma * old + (1 - gamma) * new

    return fn


class Tracker:
    def __init__(
            self,
            model_spec: dict = DEFAULT_MODEL_SPEC,
            dt: float = 1 / 24,
            x0: Optional[Vector] = None,
            box0: Optional[Box] = None,
            max_staleness: float = 12.0,
            smooth_score_gamma: float = 0.8,
            smooth_feature_gamma: float = 0.9):
        self.id = str(uuid.uuid4())
        self.model_spec = model_spec

        self.steps_alive = 1
        self.steps_positive = 1
        self.staleness = 0.0
        self.max_staleness = max_staleness

        self.update_score_fn = exponential_moving_average_fn(smooth_score_gamma)
        self.update_feature_fn = exponential_moving_average_fn(smooth_feature_gamma)

        self.score = None
        self.feature = None

        logger.debug(
            'creating new object tracker with %s and id %s' % (self.model_spec, self.id))

        self.model = Model(dt=dt, **self.model_spec)

        if x0 is None:
            x0 = self.model.box_to_x(box0)

        self._tracker = get_single_object_tracker(model=self.model, x0=x0)

    def predict(self):
        self.steps_alive += 1
        self._tracker.predict()

    def update(self, detection: Detection):
        self.steps_positive += 1

        # KF tracker update for position and size
        z = self.model.box_to_z(detection.box)
        self._tracker.update(z)

        self.score = self.update_score_fn(old=self.score, new=detection.score)
        self.feature = self.update_feature_fn(old=self.feature, new=detection.feature)

        # reduce the staleness of a tracker, faster than growth rate
        self.unstale(rate=3)

    def stale(self, rate: float = 1.0):
        self.staleness += 1
        return self.staleness

    def unstale(self, rate: float = 2.0):
        self.staleness = max(0, self.staleness - rate)
        return self.staleness

    @property
    def is_stale(self) -> bool:
        return self.staleness >= self.max_staleness

    @property
    def is_invalid(self) -> bool:
        try:
            has_nans = any(np.isnan(self._tracker.x))
            return has_nans
        except Exception as e:
            logger.warning('invalid tracker, exception: %s' % str(e))
            return True

    @property
    def box(self):
        return self.model.x_to_box(self._tracker.x)

    def __repr__(self):
        fmt = "box: %s\tstaleness: %d"
        return fmt % (self.box, self.staleness)


""" assignment cost calculation & matching methods """


def match_by_cost_matrix(trackers: Sequence[Tracker],
                         detections: Sequence[Detection],
                         min_iou: float = 0.1,
                         **kwargs) -> np.ndarray:
    if len(trackers) == 0 or len(detections) == 0:
        return []

    cost_mat, iou_mat = cost_matrix_iou_feature(trackers, detections, **kwargs)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_mat)

    # filter out low IOU matches
    ret = [[r, c] for r, c in zip(row_ind, col_ind) if iou_mat[r, c] >= min_iou]
    return np.array(ret)


def _sequence_has_none(seq: Sequence[Any]) -> Sequence[Any]:
    return any(r is None for r in seq)


def cost_matrix_iou_feature(trackers: Sequence[Tracker],
                            detections: Sequence[Detection],
                            feature_similarity_fn=angular_similarity,
                            feature_similarity_beta: float = None):

    # boxes
    b1 = np.array([t.box for t in trackers])
    b2 = np.array([d.box for d in detections])

    # box iou
    inferred_dim = int(len(b1[0]) / 2)
    iou_mat = calculate_iou(b1, b2, dim=inferred_dim)

    # feature similarity
    if feature_similarity_beta is not None:
        # get features
        f1 = [t.feature for t in trackers]
        f2 = [d.feature for d in detections]

        if _sequence_has_none(f1) or _sequence_has_none(f2):
            # fallback to pure IOU due to missing features
            apt_mat = iou_mat
        else:
            sim_mat = feature_similarity_fn(f1, f2)
            sim_mat = feature_similarity_beta + (1 - feature_similarity_beta) * sim_mat

            # combined aptitude
            apt_mat = np.multiply(iou_mat, sim_mat)
    else:
        apt_mat = iou_mat

    cost_mat = -1.0 * apt_mat
    return cost_mat, iou_mat


class MatchingFunction:
    def __call__(self,
                 trackers: Sequence[Tracker],
                 detections: Sequence[Detection]) -> np.ndarray:
        raise NotImplementedError()


class BasicMatchingFunction(MatchingFunction):
    """ class implements the most basic matching function, taking
    detections boxes and optional feature similarity into account """

    def __init__(self, min_iou: float = 0.1,
                 feature_similarity_fn=angular_similarity,
                 feature_similarity_beta: Optional[float] = None) -> None:

        self.min_iou = min_iou
        self.feature_similarity_fn = feature_similarity_fn
        self.feature_similarity_beta = feature_similarity_beta

    def __call__(self,
                 trackers: Sequence[Tracker],
                 detections: Sequence[Detection]) -> np.ndarray:
        return match_by_cost_matrix(
            trackers, detections,
            self.min_iou,
            feature_similarity_fn=self.feature_similarity_fn,
            feature_similarity_beta=self.feature_similarity_beta)


class MultiObjectTracker:
    def __init__(self, dt: float,
                 model_spec: Union[str, dict] = DEFAULT_MODEL_SPEC,
                 matching_fn: Optional[MatchingFunction] = None,
                 tracker_kwargs: dict = None,
                 matching_fn_kwargs: dict = None,
                 active_tracks_kwargs: dict = None) -> None:
        """
            model_spec specifies the dimension and order for position and size of the object
            matching_fn determines the strategy on which the trackers and detections are assigned.

            tracker_kwargs are passed to each single object tracker
            active_tracks_kwargs limits surfacing of fresh/fading out tracks
        """

        self.dt = dt
        self.trackers = []

        if isinstance(model_spec, dict):
            self.model_spec = model_spec
        elif isinstance(model_spec, str) and model_spec in ModelPreset.__members__:
            self.model_spec = ModelPreset[model_spec].value
        else:
            raise NotImplementedError('unsupported motion model %s' % str(model_spec))
        logger.debug('using model spec: %s' % str(self.model_spec))

        self.matching_fn = matching_fn
        self.matching_fn_kwargs = matching_fn_kwargs if matching_fn_kwargs is not None else {}
        if self.matching_fn is None:
            self.matching_fn = BasicMatchingFunction(**self.matching_fn_kwargs)

        # kwargs to be passed to each single object tracker
        self.tracker_kwargs = tracker_kwargs if tracker_kwargs is not None else {}
        logger.debug('using tracker_kwargs: %s' % str(self.tracker_kwargs))

        # kwargs to be used when self.step returns active tracks
        self.active_tracks_kwargs = active_tracks_kwargs if active_tracks_kwargs is not None else {}
        logger.debug('using active_tracks_kwargs: %s' % str(self.active_tracks_kwargs))

    def active_tracks(self,
                      max_staleness_to_positive_ratio: float = 3.0,
                      max_staleness: float = 999,
                      min_steps_alive: int = -1) -> Sequence[Track]:
        """ returns all active tracks after optional filtering by tracker steps count and staleness """

        tracks = []
        for tracker in self.trackers:
            cond1 = tracker.staleness / tracker.steps_positive < max_staleness_to_positive_ratio  # early stage
            cond2 = tracker.staleness < max_staleness
            cond3 = tracker.steps_alive >= min_steps_alive
            if cond1 and cond2 and cond3:
                tracks.append(Track(id=tracker.id, box=tracker.box, score=tracker.score))

        logger.debug('active/all tracks: %d/%d' % (len(self.trackers), len(tracks)))
        return tracks

    def cleanup_trackers(self) -> None:
        count_before = len(self.trackers)
        self.trackers = [t for t in self.trackers if not (t.is_stale or t.is_invalid)]
        count_after = len(self.trackers)
        logger.debug('deleted %s/%s trackers' % (count_before - count_after, count_before))

    def step(self, detections: Sequence[Detection]) -> Sequence[Track]:
        """ the method matches the new detections with existing trackers,
        creates new trackers if necessary and performs the cleanup.
        Returns the active tracks after active filtering applied """

        # filter out empty detections
        detections = [det for det in detections if det.box is not None]

        logger.debug('step with %d detections' % len(detections))
        matches = self.matching_fn(self.trackers, detections)
        logger.debug('matched %d pairs' % len(matches))

        # all trackers: predict
        for t in self.trackers:
            t.predict()

        # assigned trackers: correct
        for match in matches:
            track_idx, det_idx = match[0], match[1]
            self.trackers[track_idx].update(detection=detections[det_idx])

        # not assigned detections: create new trackers POF
        assigned_det_idxs = set(matches[:, 1]) if len(matches) > 0 else []
        for det_idx in set(range(len(detections))).difference(assigned_det_idxs):
            tracker = Tracker(box0=detections[det_idx].box,
                              model_spec=self.model_spec,
                              **self.tracker_kwargs)
            self.trackers.append(tracker)

        # unassigned trackers
        assigned_track_idxs = set(matches[:, 0]) if len(matches) > 0 else []
        for track_idx in set(range(len(self.trackers))).difference(assigned_track_idxs):
            self.trackers[track_idx].stale()

        # cleanup dead trackers
        self.cleanup_trackers()

        return self.active_tracks(**self.active_tracks_kwargs)
