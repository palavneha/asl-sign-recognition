import cv2
import numpy as np

IMG_SIZE = 128

# Removed CLAHE as it exaggerates background noise.

def preprocess_image(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)

    # Read as grayscale directly since hand tracking is temporarily disabled
    gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None

    img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # Apply blur to smooth out noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Canny Edge Detection to get outlines
    edges = cv2.Canny(blurred, 50, 150)

    # Normalize
    edges = edges.astype("float32") / 255.0

    # Expand dimensions for CNN
    edges = np.expand_dims(edges, axis=-1)
    edges = np.expand_dims(edges, axis=0)

    return edges
