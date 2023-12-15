if __name__ == "__main__":
    import cv2
    from Yolo import Yolo
    from TextOCR import Reader
    from matplotlib import pyplot as plt

    yolo = Yolo()

    # The first argument is the path to the image
    import sys
    if len(sys.argv) != 2:
        print("Usage: python singleImage.py <path_to_image>")
        exit(1)
    image_path = sys.argv[1]

    reader = Reader()

    img = cv2.imread(image_path)

    bboxes = yolo.find_bboxes(img)
    print(bboxes)

    for box in bboxes:
        x, y, w, h = box
        
        if w > 40:
            cv2.putText(img, reader.read(
                img[y:y+h, x:x+w]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(img[:, :, ::-1])
    plt.show()
