import os
import sys
import numpy as np
import tensorflow as tf

# Fix Unicode emoji printing on Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = r"D:\asl-sign-recognition"
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_asl_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.npy")

model = tf.keras.models.load_model(MODEL_PATH)
labels = np.load(LABELS_PATH, allow_pickle=True)

print("[OK] Model loaded successfully")


def predict(img):
    preds = model.predict(img, verbose=0)[0]
    idx = np.argmax(preds)
    confidence = float(preds[idx])
    return labels[idx], confidence
