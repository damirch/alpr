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
    from TextOCR import Reader
    from matplotlib import pyplot as plt

    yolo = Yolo()

    # if there is one argument, use it as use_tesseract
    import sys
    if len(sys.argv) == 2:
        reader = Reader(use_tesseract=sys.argv[1])
    else:
        reader = Reader()

    img = cv2.imread(random_image_path())
    
    bboxes = yolo.find_bboxes(img)
    print(bboxes)

    for box in bboxes:
        x, y, w, h = box
        plt.imshow(img[y:y+h, x:x+w, ::-1])
        plt.show()
        if w > 40:
            cv2.putText(img, reader.read(img[y:y+h, x:x+w]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(img[:, :, ::-1])
    plt.show()