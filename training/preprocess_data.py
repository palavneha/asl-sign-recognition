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
IMG_SIZE = 64
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

# CLAHE (keep for better contrast)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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

        # Contrast enhancement
        gray = clahe.apply(gray)

        # Normalize
        gray = gray.astype("float32") / 255.0

        # Skip low-quality images
        if gray.std() < 0.03:
            continue

        # Add channel dimension → (64,64,1)
        gray = np.expand_dims(gray, axis=-1)

        X.append(gray)
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

# Save
np.save(os.path.join(SAVE_DIR, "X.npy"), X)
np.save(os.path.join(SAVE_DIR, "y.npy"), y)
np.save(os.path.join(SAVE_DIR, "labels.npy"), labels)

print("\n✅ Fast preprocessing completed!")


# In[ ]:




