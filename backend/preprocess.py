import cv2
import numpy as np

IMG_SIZE = 64

# SAME CLAHE AS TRAINING
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_image(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # 🔥 ADD THIS (IMPORTANT)
    img = clahe.apply(img)

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    return img
