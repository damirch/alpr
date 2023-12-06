import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import os

YOLO_PATH = "../yolo_weights/eu_dataset_256_160_extended/"


class Yolo:
    def __init__(
        self,
        weight_path=os.path.join(YOLO_PATH, "yolov4-tiny.weights"),
        config_path=os.path.join(YOLO_PATH, "yolov4-tiny.cfg"),
        dims=(256, 160),
    ):

        print("Initializing Yolo...")

        print(f"Using weights: {weight_path}")
        print(f"Using config: {config_path}")

        self.dims = dims
        print(f"Using dims: {self.dims}")

        self.net = cv.dnn.readNetFromDarknet(config_path, weight_path)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

        # determine the output layer
        self.outputLayerNames = self.net.getLayerNames()
        # print(self.outputLayerNames)
        # print(self.net.getUnconnectedOutLayers())
        self.outputLayerNames = [self.outputLayerNames[i - 1]
                                 for i in self.net.getUnconnectedOutLayers()]

        print("Yolo initialized.")

    def find_bboxes(self, raw_img):
        blob = cv.dnn.blobFromImage(
            raw_img, 1/255.0, self.dims, swapRB=True, crop=False)

        self.net.setInput(blob)

        outputs = self.net.forward(self.outputLayerNames)

        boxes = []
        confidences = []
        h, w = raw_img.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                confidence = scores[0] # 0 because there is only one class: license plate

                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))

        indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        return [boxes[i] for i in indices]
    
    def draw_bboxes_video(self, input_path: str, output_path: str):
        """Draws bounding boxes on the video and saves it to the same directory as the video. (MP4)"""
        cap = cv.VideoCapture(input_path) 
  
        output = cv.VideoWriter(output_path,
                                cv.VideoWriter_fourcc(*"mp4v"),
                                30,
                                (int(cap.get(3)), int(cap.get(4))))

        frame_counter = 0
    
        while(True): 
            ret, frame = cap.read() 
            if(ret): 
                
                # draw yolo bounding boxes
                bboxes = self.find_bboxes(frame)
                for box in bboxes:
                    x, y, w, h = box
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # writing the new frame in output 
                output.write(frame) 
            else: 
                break

            frame_counter += 1
            if frame_counter % 100 == 0:
                print(frame_counter)

        output.release() 
        cap.release() 

    @staticmethod
    def demo(image_path: str):
        yolo = Yolo()
        img = cv.imread(image_path)
        bboxes = yolo.find_bboxes(img)
        img = cv.imread(image_path)

        for i, box in enumerate(bboxes):
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        plt.imshow(img[:, :, ::-1])
        plt.show()


# example: python3 Yolo.py ../eu_dataset/images/dayride_type1_001\#t=528.jpg

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the image path as a command line argument.")
        sys.exit(1)

    image_path = sys.argv[1]
    Yolo.demo(image_path)
