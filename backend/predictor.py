import os
import numpy as np
import tensorflow as tf

# -------------------------
# Load model & labels ONCE
# -------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "asl_model_full.keras")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.npy")

model = tf.keras.models.load_model(MODEL_PATH)
labels = np.load(LABELS_PATH, allow_pickle=True)

print("✅ Model loaded successfully")

# -------------------------
# Prediction function
# -------------------------

def predict(img):
    preds = model.predict(img, verbose=0)[0]

    idx = np.argmax(preds)
    confidence= float(preds[idx])

    return labels[idx], confidence