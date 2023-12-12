from Yolo import Yolo
from TextOCR import Reader
import sys
import cv2 as cv

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please provide the video path and the target path as a command line argument.")
        sys.exit(1)

    yolo = Yolo()
    reader = Reader()

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    max_frames = int(sys.argv[3]) if len(sys.argv) == 4 else 1e100

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
            bboxes = yolo.find_bboxes(frame)
            for box in bboxes:
                x, y, w, h = box
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if w > 60:
                    try:
                        cv.putText(frame, reader.read(frame[y:y+h, x:x+w]), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                    except:
                        print("OCR failed on image.")

            # writing the new frame in output 
            output.write(frame) 
        else: 
            break

        frame_counter += 1
        if frame_counter % 100 == 0:
            print(frame_counter)

        if frame_counter >= max_frames:
            break

    output.release() 
    cap.release()