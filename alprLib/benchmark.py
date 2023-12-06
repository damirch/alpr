from Yolo import Yolo
from randomImage import random_image_path
import time
import cv2

yolo_small = Yolo("../yolo_weights/eu_dataset_256_160_extended/yolov4-tiny.weights", "../yolo_weights/eu_dataset_256_160_extended/yolov4-tiny.cfg", (256, 160))
yolo_large = Yolo("../yolo_weights/eu_dataset_512_320/yolov4-tiny.weights", "../yolo_weights/eu_dataset_512_320/yolov4-tiny.cfg", (512, 320))

nb_runs = 200
inference_times_small = []
inference_times_large = []

for i in range(nb_runs):
    img = cv2.imread(random_image_path())

    start = time.time()
    bboxes = yolo_small.find_bboxes(img)
    end = time.time()
    inference_times_small.append(end - start)

    start = time.time()
    bboxes = yolo_large.find_bboxes(img)
    end = time.time()
    inference_times_large.append(end - start)

print(f"Average inference time for small model: {sum(inference_times_small) / len(inference_times_small)}")
print(f"Average inference time for large model: {sum(inference_times_large) / len(inference_times_large)}")