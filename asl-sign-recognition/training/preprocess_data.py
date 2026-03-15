#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =====================================
# ASL DATA PREPROCESSING (FAST VERSION)
# =====================================

import os
import cv2
import numpy as np

# -------- SETTINGS --------
DATASET_PATH = r"C:\Users\Dell\Desktop\asl-sign-recognition\datasets"
IMG_SIZE = 128
SAVE_DIR = r"C:\Users\Dell\Desktop\asl-sign-recognition\models"
# --------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

# Get class folders
labels = sorted([
    f for f in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, f))
])

label_map = {label: idx for idx, label in enumerate(labels)}

print("Classes:", labels)
print("Number of classes:", len(labels))

X, y = [], []

# Removed CLAHE as it exaggerates background noise.
count = 0

for class_name in labels:
    class_path = os.path.join(DATASET_PATH, class_name)
    label = label_map[class_name]

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        # 🔥 Read directly as grayscale (FASTER)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        # Resize
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

        # Apply slight blur to remove tiny background noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny Edge Detection (converts image to pure edge outlines)
        edges = cv2.Canny(blurred, 50, 150)

        # Normalize edge map (Canny outputs 0 or 255)
        edges = edges.astype("float32") / 255.0

        # Optional: Skip blank edge maps (nothing detected)
        if edges.sum() < 100:
            continue

        # Add channel dimension → (128,128,1)
        edges = np.expand_dims(edges, axis=-1)

        X.append(edges)
        y.append(label)

        count += 1
        if count % 500 == 0:
            print(f"Processed {count} images...")

# Convert to numpy
X = np.array(X)
y = np.array(y)

print("\nFinal shapes:")
print("X:", X.shape)
print("y:", y.shape)

# Save together in compressed format
print("\nSaving arrays compressed to disk...")
np.savez_compressed(
    os.path.join(SAVE_DIR, "dataset.npz"), 
    X=X, 
    y=y, 
    labels=labels
)

print("\n✅ Fast preprocessing completed!")


# In[ ]:




