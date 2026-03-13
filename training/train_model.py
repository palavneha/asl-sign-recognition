# =====================================
# ASL CNN MODEL TRAINING (NO AUGMENTATION)
# =====================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, Input
)
from sklearn.model_selection import train_test_split
from collections import Counter

# -------- PATHS --------
BASE_DIR = r"C:\Users\Dell\Desktop\asl-sign-recognition"
MODEL_DIR = os.path.join(BASE_DIR, "models")
# -----------------------

# ======================
# LOAD DATA
# ======================
X = np.load(os.path.join(MODEL_DIR, "X.npy"))
y = np.load(os.path.join(MODEL_DIR, "y.npy"))
labels = np.load(os.path.join(MODEL_DIR, "labels.npy"), allow_pickle=True)

print("Data loaded:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", labels)
print("Class distribution:", Counter(y))


# ======================
# TRAIN TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)


# ======================
# CNN MODEL (CLEAN)
# ======================
model = Sequential([

    Input(shape=(64, 64, 1)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(len(labels), activation='softmax')
])


# ======================
# COMPILE
# ======================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ======================
# TRAIN
# ======================
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    shuffle=True
)


# ======================
# SAVE
# ======================
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, "asl_model_full.keras"))

print("✅ Model training completed and saved")

