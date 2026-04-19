import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
import urllib.request
import os

IMG_SIZE = 128

# -------------------------------
# Download model if not present
# -------------------------------
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        MODEL_PATH,
    )
    print("Downloaded!")

# -------------------------------
# Initialize detector
# -------------------------------
options = HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
)

detector = HandLandmarker.create_from_options(options)


# -------------------------------
# Preprocessing function
# -------------------------------
def preprocess_image(img_bytes, save_steps=False):
    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr is None:
        print("Error: Image not loaded")
        return None

    h, w, _ = bgr.shape

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        x_coords = [int(lm.x * w) for lm in landmarks]
        y_coords = [int(lm.y * h) for lm in landmarks]

        x_min = max(min(x_coords) - 20, 0)
        x_max = min(max(x_coords) + 20, w)
        y_min = max(min(y_coords) - 20, 0)
        y_max = min(max(y_coords) + 20, h)

        hand_crop = bgr[y_min:y_max, x_min:x_max]

        if hand_crop.size == 0:
            print("Empty crop")
            return None
    else:
        print("No hand detected")
        return None

    # -------------------------------
    # Preprocessing steps
    # -------------------------------
    gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    # -------------------------------
    # Save images for poster
    # -------------------------------
    if save_steps:
        cv2.imwrite("step1_hand_crop.jpg", hand_crop)
        cv2.imwrite("step2_grayscale.jpg", gray)
        cv2.imwrite("step3_edges.jpg", edges)
        print("Saved preprocessing images!")

    # -------------------------------
    # Normalize for model
    # -------------------------------
    edges = edges.astype("float32") / 255.0
    edges = np.expand_dims(edges, axis=-1)
    edges = np.expand_dims(edges, axis=0)

    return edges


# -------------------------------
# MAIN TEST RUN
# -------------------------------
if __name__ == "__main__":
    image_path = "sample.jpg"  # <-- put your image here

    if not os.path.exists(image_path):
        print("Put a sample image named 'sample.jpg' in this folder.")
    else:
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        preprocess_image(img_bytes, save_steps=True)