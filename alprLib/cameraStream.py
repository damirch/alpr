import numpy as np
import cv2
import datetime

from Yolo import Yolo
from TextOCR import Reader
    
def stream(saveToDisk=False):
    # create display window
    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)

    # initialize webcam capture object
    cap = cv2.VideoCapture(0)

    # retrieve properties of the capture object
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_sleep = int(1000 / cap_fps)
    print('* Capture width:', cap_width)
    print('* Capture height:', cap_height)
    print('* Capture FPS:', cap_fps, 'ideal wait time between frames:', fps_sleep, 'ms')

    # initialize time and frame count variables
    last_time = datetime.datetime.now()
    frames = 0


    output = cv2.VideoWriter("outputCam.mp4",
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            30,
                            (int(cap.get(3)), int(cap.get(4))))


    yolo = Yolo()
    reader = Reader()

    # main loop: retrieves and displays a frame from the camera
    while (True):
        # blocks until the entire frame is read
        success, img = cap.read()
        frames += 1

        # compute fps: current_time - last_time
        delta_time = datetime.datetime.now() - last_time
        elapsed_time = delta_time.total_seconds()
        cur_fps = np.around(frames / elapsed_time, 1)

        # ALPR
        bboxes = yolo.find_bboxes(img)
        for box in bboxes:
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if w > 60:
                try:
                    cv2.putText(img, reader.read(img[y:y+h, x:x+w]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                except:
                    print("OCR failed on image.")

        # draw FPS text and display image
        cv2.putText(img, 'FPS: ' + str(cur_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("webcam", img)

        # write to disk
        if saveToDisk:
            output.write(img)

        # wait 1ms for ESC to be pressed
        key = cv2.waitKey(1)
        if (key == 27):
            break

    # release resources
    cv2.destroyAllWindows()
    cap.release()

    if saveToDisk:
        output.release()


if __name__ == "__main__":
    # first argument is saveToDisk
    import sys
    if len(sys.argv) == 2:
        stream(saveToDisk=sys.argv[1])
    else:
        stream()
