# =====================================
# ASL CNN MODEL TRAINING (NO AUGMENTATION)
# =====================================

import os
# 🔥 MUST BE SET BEFORE TENSORFLOW IMPORT TO PREVENT WINDOWS CPU CRASH 🔥
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, Input, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from collections import Counter

# -------- PATHS --------
BASE_DIR = r"C:\Users\Dell\Desktop\asl-sign-recognition"
MODEL_DIR = os.path.join(BASE_DIR, "models")
# -----------------------

# ======================
# LOAD DATA
# ======================
dataset = np.load(os.path.join(MODEL_DIR, "dataset.npz"), allow_pickle=True)
X = dataset['X']
y = dataset['y']
labels = dataset['labels']

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
# DATA AUGMENTATION
# ======================
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,  # ASL uses specific hands, flipping can change meaning
    fill_mode='nearest'
)

# ======================
# CNN MODEL (CLEAN)
# ======================
model = Sequential([

    Input(shape=(128, 128, 1)),

    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
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
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
    ModelCheckpoint(os.path.join(MODEL_DIR, "best_asl_model.keras"), save_best_only=True, monitor='val_accuracy')
]

# Reduced batch size to 16 to prevent memory spikes on CPU
batch_size = 16
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    epochs=40, # increased epochs since we use early stopping
    validation_data=(X_test, y_test),
    steps_per_epoch=max(1, len(X_train) // batch_size),
    callbacks=callbacks,
    shuffle=True
)


# ======================
# SAVE
# ======================
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, "asl_model_full.keras"))

print("✅ Model training completed and saved")

