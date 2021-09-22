import time

import cv2
from motpy import ModelPreset, MultiObjectTracker
from motpy.core import setup_logger
from motpy.testing_viz import draw_rectangle, draw_track, image_generator
from motpy.utils import ensure_packages_installed

ensure_packages_installed(['cv2'])


logger = setup_logger(__name__, 'DEBUG', is_main=True)


def demo_tracking_visualization(
        model_spec=ModelPreset.constant_acceleration_and_static_box_size_2d.value,
        num_steps: int = 1000,
        num_objects: int = 20):
    gen = image_generator(
        num_steps=num_steps,
        num_objects=num_objects,
        max_omega=0.03,
        miss_prob=0.33,
        disappear_prob=0.00,
        det_err_sigma=3.33)

    dt = 1 / 24
    tracker = MultiObjectTracker(
        dt=dt,
        model_spec=model_spec,
        active_tracks_kwargs={'min_steps_alive': 2, 'max_staleness': 6},
        tracker_kwargs={'max_staleness': 12})

    for _ in range(num_steps):
        img, _, detections = next(gen)
        detections = [d for d in detections if d.box is not None]

        t0 = time.time()
        active_tracks = tracker.step(detections=detections)
        elapsed = (time.time() - t0) * 1000.
        logger.debug(f'tracking elapsed time: {elapsed:.3f} ms')

        for track in active_tracks:
            draw_track(img, track)

        for det in detections:
            draw_rectangle(img, det.box, color=(10, 220, 20), thickness=1)

        cv2.imshow('preview', img)
        # stop the demo by pressing q
        wait_ms = int(1000 * dt)
        c = cv2.waitKey(wait_ms)
        if c == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_tracking_visualization()
