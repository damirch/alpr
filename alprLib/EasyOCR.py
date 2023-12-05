import easyocr
import numpy as np

class Reader:
    def __init__(self) -> None:
        print("Initializing EasyOCR...")
        self.reader = easyocr.Reader(['en'])
        self.allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        print("EasyOCR initialized.")

    def read(self, img: np.ndarray) -> str:
        result = self.reader.recognize(img, batch_size=5, allowlist=self.allowed_chars, detail=0)
        if len(result) == 0:
            return ""
        else:
            return result[0]