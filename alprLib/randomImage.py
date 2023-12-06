import random as rd
import os

IMAGE_FOLDER = "../eu_dataset/images"

def random_image_path():
    """Returns a random image from the dataset."""
    files = os.listdir(IMAGE_FOLDER)
    file = rd.choice(files)
    return os.path.join(IMAGE_FOLDER, file)

if __name__ == "__main__":
    import cv2
    from Yolo import Yolo
    from matplotlib import pyplot as plt

    yolo = Yolo()
    img = cv2.imread(random_image_path())
    
    bboxes = yolo.find_bboxes(img)
    print(bboxes)

    for box in bboxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(img[:, :, ::-1])
    plt.show()