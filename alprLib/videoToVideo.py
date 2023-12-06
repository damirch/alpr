from Yolo import Yolo
import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please provide the video path and the target path as a command line argument.")
        sys.exit(1)

    yolo = Yolo()
    video_path = sys.argv[1]
    target_path = sys.argv[2]

    yolo.draw_bboxes_video(video_path, target_path)