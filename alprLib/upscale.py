import cv2
from matplotlib import pyplot as plt

def upscale(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("./super_resolution/EDSR_x4.pb")
    sr.setModel("edsr", 4)

    return sr.upsample(img)

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Please provide the input path as a command line argument.")
        sys.exit(1)

    input_path = sys.argv[1]
    
    img = cv2.imread(input_path)
    upscaled = upscale(img)
    plt.imshow(upscaled)
    plt.show()