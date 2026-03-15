import sys
import os
import cv2
import numpy as np

# Load the preprocessing script from backend
sys.path.append(r"d:\asl-sign-recognition\asl-sign-recognition\backend")
from preprocess import preprocess_image
from predictor import predict

DATASET_PATH = r"C:\Users\Dell\Desktop\asl-sign-recognition\datasets"

if not os.path.exists(DATASET_PATH):
    print(f"Dataset not found at {DATASET_PATH}")
    sys.exit(1)

# Pick a few classes and test images
classes = ["A", "B", "C"]
for cls in classes:
    class_dir = os.path.join(DATASET_PATH, cls)
    if not os.path.exists(class_dir):
         continue
    files = os.listdir(class_dir)[:3]
    for file in files:
         img_path = os.path.join(class_dir, file)
         with open(img_path, "rb") as f:
             img_bytes = f.read()
         
         processed = preprocess_image(img_bytes)
         
         if processed is None:
             print(f"[{cls} - {file}] NO HAND DETECTED")
             continue
         
         label, conf = predict(processed)
         print(f"[{cls} - {file}] PREDICTED: {label} (Conf: {conf:.2f})")

