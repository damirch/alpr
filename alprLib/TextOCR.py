import easyocr
import numpy as np
import cv2
import pytesseract
from matplotlib import pyplot as plt

class Reader:
    def __init__(self, use_tesseract: bool = False) -> None:
        if not use_tesseract:
            print("Initializing EasyOCR...")
        else:
            print("using tesseract OCR...")
        self.use_tesseract = use_tesseract
        self.reader = easyocr.Reader(['en'])
        self.allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "

        if not use_tesseract:
            print("EasyOCR initialized.")
        else:
            print("tesseract OCR initialized.")

    def read(self, img: np.ndarray) -> str:
        # preprocess
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # histogram equalization
        gray = cv2.equalizeHist(gray)

        # sharpen
        #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        #gray = cv2.filter2D(gray, -1, 0.2 * kernel)

        # max width is 128, but keep aspect ratio
        #scale = 128 / gray.shape[1]
        #gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)

        # Gaussian filter
        #gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # threshold
        #gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        plt.imshow(gray, cmap="gray")
        plt.show()

        if not self.use_tesseract:
            results = self.reader.readtext(gray, batch_size=5, allowlist=self.allowed_chars)
            # fatser, but less accurate
            #result = self.reader.recognize(gray, batch_size=5, allowlist=self.allowed_chars, detail=0)
        else:
            return pytesseract.image_to_string(gray, config=r'--oem 3 --psm 6')

        
        return "".join([result[1] for result in results])