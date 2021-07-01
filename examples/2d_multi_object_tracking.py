import cv2
from motpy import ModelPreset, MultiObjectTracker
from motpy.core import setup_logger
from motpy.testing_viz import draw_rectangle, draw_text, image_generator

logger = setup_logger(__name__, is_main=True)


def to_int_tuple(v):
    return tuple([int(el) for el in v])

def demo_tracking_visualization(num_steps: int = 1000, num_objects: int = 10):
    gen = image_generator(
        num_steps=num_steps,
        num_objects=num_objects,
        max_omega=0.03,
        miss_prob=0.33,
        disappear_prob=0.00,
        det_err_sigma=3.33)

    # constant velocity model
    model_spec = {'order_pos': 1, 'dim_pos': 2,
                  'order_size': 0, 'dim_size': 2,
                  'q_var_pos': 5000., 'r_var_pos': 0.1}

    dt = 1 / 24
    tracker = MultiObjectTracker(
        dt=dt,
        model_spec=model_spec,
        active_tracks_kwargs={'min_steps_alive': 2, 'max_staleness': 6},
        tracker_kwargs={'max_staleness': 12})

    for _ in range(num_steps):
        img, _, detections = next(gen)

        detections = [d for d in detections if d.box is not None]
        active_tracks = tracker.step(detections=detections)

        for track in active_tracks:
            score = track.score if track.score is not None else -1
            img = draw_rectangle(img, track.box, color=(10, 10, 220), thickness=5)
            img = draw_text(img, f'{track.id[:8]}... ({score:.2f})', above_box=track.box)

            # draw velocity
            ptc = (int((track.box[0] + track.box[2]) / 2), int((track.box[1] + track.box[3]) / 2))
            cv2.line(img, ptc, (ptc[0] + int(track.v[0]), ptc[1]), (255,0,0), 1) # horizontal velocity
            cv2.line(img, ptc, (ptc[0], ptc[1] + int(track.v[1])), (255,0,0), 1) # vettical velocity
            cv2.line(img, ptc, (ptc[0] + int(track.v[0]), ptc[1] + int(track.v[1])), (0,255,0), 2) # combined

        for det in detections:
            img = draw_rectangle(img, det.box, color=(10, 220, 20), thickness=1)

        cv2.imshow('preview', img)
        # stop the demo by pressing q
        wait_ms = int(1000 * dt)
        c = cv2.waitKey(wait_ms)
        if c == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_tracking_visualization()
