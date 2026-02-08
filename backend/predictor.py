import tensorflow as tf
import numpy as np

MODEL_PATH = r"C:\Users\Dell\Desktop\asl-sign-recognition\models\asl_model_full.keras"
LABELS_PATH = r"C:\Users\Dell\Desktop\asl-sign-recognition\models\labels.npy"

print("Loading model:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)

def predict(img):
    preds = model.predict(img, verbose=0)
    idx = np.argmax(preds)
    return labels[idx], float(preds[0][idx])
