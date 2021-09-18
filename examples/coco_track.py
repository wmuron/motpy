import os
import time

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from motpy import Detection, MultiObjectTracker
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model.eval()
from motpy.core import setup_logger
from motpy.detector import BaseObjectDetector
from motpy.testing_viz import draw_detection, draw_track

logger = setup_logger(__name__, 'DEBUG', is_main=True)


video_path = os.path.expanduser('~/datasets/videos_raw/yt/ny4k_queens_sunset_4m.mp4')
cap = cv2.VideoCapture(video_path)

# frame_tensors = torchvision.io.read_video(video_path)

# for frame in frame_tensors:
# print(frame)

# from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
# this will help us create a different color for each class
# COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
# define the torchvision image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])


def get_model(device):
    # load the model
    # model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # load the model onto the computation device
    model = model.eval().to(device)
    return model


def predict(image, model, device, detection_threshold):
    """
    Predict the output of an image after forward pass through
    the model and return the bounding boxes, class names, and 
    class labels. 
    """
    # transform the image to tensor
    image = transform(image).to(device)
    # add a batch dimension
    image = image.unsqueeze(0)
    # get the predictions on the image
    with torch.no_grad():
        outputs = model(image)
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    labels = outputs[0]['labels'][:len(boxes)]
    # scores = outputs[0]['scores'][:len(boxes)]
    # get all the predicited class names
    # pred_classes = [str(i) for i in labels.cpu().numpy()]
    return boxes, labels


model = get_model('cuda')


tracker = MultiObjectTracker(
    dt=1 / 15., tracker_kwargs={'max_staleness': 15},
    model_spec=None,
    matching_fn_kwargs={'min_iou': 0.25})

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, fx=0.75, fy=0.75, dsize=None, interpolation=cv2.INTER_AREA)

    t0 = time.time()
    boxes, labels = predict(frame, model, 'cuda', 0.5)
    elapsed = (time.time() - t0) * 1000.
    logger.debug(f'inference time: {elapsed:.3f} ms')

    detections = []
    for b, l in zip(boxes, labels):
        detections.append(Detection(box=b, score=1.0))

    for det in detections:
        draw_detection(frame, det)

    t0 = time.time()
    active_tracks = tracker.step(detections=detections)
    elapsed = (time.time() - t0) * 1000.
    logger.debug(f'tracking time: {elapsed:.3f} ms')

    for track in active_tracks:
        draw_track(frame, track, thickness=2, text_at_bottom=True)

    # print(detections)
    # print(frame)

    # inp = torch.Tensor([frame])
    # inp = inp.permute(0, 3, 1, 2)

    # inp = inp.transpose((2, 0, 1))
    # inp = np.expand_dims(img, 0)
    # For training
    # images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
    # labels = torch.randint(1, 91, (4, 11))
    # images = list(image for image in images)
    # targets = []
    # for i in range(len(images)):
    #     d = {}
    #     d['boxes'] = boxes[i]
    #     d['labels'] = labels[i]
    #     targets.append(d)

    # output = model(images, targets)
    # For inference

    # x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    # with torch.no_grad():
    #     predictions = model(inp)

    # print(predictions)

    cv2.imshow('frame', frame)
    _ = cv2.waitKey(10)

    # optionally, if you want to export the model to ONNX:
    # torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)
