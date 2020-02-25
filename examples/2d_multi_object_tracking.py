
import cv2

import motpy
from motpy import MultiObjectTracker, ModelPreset
from motpy.testing_viz import draw_rectangle, draw_text, image_generator

motpy.set_log_level('DEBUG')


def demo_tracking_visualization(num_steps: int = 1000, num_objects: int = 10):
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
        model_spec=ModelPreset.constant_acceleration_and_static_box_size_2d.value,
        active_tracks_kwargs={'min_steps_alive': 2, 'max_staleness': 6},
        tracker_kwargs={'max_staleness': 12})

    for _ in range(num_steps):
        img, _, detections = next(gen)

        detections = [d for d in detections if d.box is not None]
        active_tracks = tracker.step(detections=detections)

        for track in active_tracks:
            img = draw_rectangle(img, track.box, color=(10, 10, 220), thickness=5)
            img = draw_text(img, track.id, above_box=track.box)

        for det in detections:
            img = draw_rectangle(img, det.box, color=(10, 220, 20), thickness=2)

        cv2.imshow('preview', img)
        # stop the demo by pressing q
        wait_ms = int(1000 * dt)
        c = cv2.waitKey(wait_ms)
        if c == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_tracking_visualization()
