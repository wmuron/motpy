import os
from typing import Sequence
from urllib.request import urlretrieve

import cv2
from motpy import Detection, MultiObjectTracker, NpImage, Box, Track
from motpy.core import setup_logger
from motpy.detector import BaseObjectDetector
from motpy.testing_viz import draw_detection, draw_track

from motpy.testing import CANVAS_SIZE, data_generator

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes

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

class motpy2darknet(Node):
    def __init__(self):

        ## By run() function --------------------------------
        self.model_spec = {'order_pos': 1, 'dim_pos': 2,
                            'order_size': 0, 'dim_size': 2,
                            'q_var_pos': 5000., 'r_var_pos': 0.1}

        self.dt = 1 / 15.0  # assume 15 fps
        self.tracker = MultiObjectTracker(dt=self.dt, model_spec=self.model_spec)

        self.motpy_detector = FaceDetector()

        ## RCLPY 
        super().__init__('motpy_ros')
        self.pub = self.create_publisher(BoundingBoxes,"bounding_boxes", 1)


        self.sub = self.create_subscription(Image,"color/image_raw",self.process_image_ros2,1)
        self.bridge = CvBridge()

    def create_d_msgs_box(self, track):
        one_box = BoundingBox()

        one_box.id = int(track.id[:3], 16)
        one_box.class_id = "face"
        one_box.probability = float(track.score)
        one_box.xmin = int(track.box[0])
        one_box.ymin = int(track.box[1])
        one_box.xmax = int(track.box[2])
        one_box.ymax = int(track.box[3])

        return one_box

    def publish_d_msgs(self, tracks, img_msg):
        
        boxes = BoundingBoxes()
        boxes.header = img_msg.header
        
        for track in tracks:
            boxes.bounding_boxes.append(self.create_d_msgs_box(track))

        print("boxes--------------------")
        for box_print in boxes.bounding_boxes:
            print(box_print)
        print("\n\n")
        
        self.pub.publish(boxes)

    def process_image_ros2(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg,"bgr8")

            frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)

            # # run face detector on current frame
            
            detections = self.motpy_detector.process_image(frame)

            self.tracker.step(detections)
            tracks = self.tracker.active_tracks(min_steps_alive=3)

            self.publish_d_msgs(tracks, msg)

            # preview the boxes on frame----------------------------------------
            for det in detections:
                draw_detection(frame, det)

            for track in tracks:
                draw_track(frame, track)

            cv2.imshow('frame', frame)
            if cv2.waitKey(int(1000 * self.dt)) & 0xFF == ord('q'):
                pass

        except Exception as err:
            print(err)

        pass

def ros_main(args = None):
    
    rclpy.init(args=args)

    motpy2darknet_class = motpy2darknet()
    rclpy.spin(motpy2darknet_class)

    motpy2darknet_class.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    ros_main()