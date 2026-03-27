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

# Download hand landmarker model if not present
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        MODEL_PATH,
    )
    print("Downloaded!")

options = HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
)
detector = HandLandmarker.create_from_options(options)


def preprocess_image(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None

    h, w, _ = bgr.shape
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
            return None
    else:
        return None

    gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = edges.astype("float32") / 255.0
    edges = np.expand_dims(edges, axis=-1)
    edges = np.expand_dims(edges, axis=0)

    return edges
