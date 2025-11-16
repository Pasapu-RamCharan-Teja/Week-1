# -*- coding: utf-8 -*-
"""Garbage Classification (Rewritten Version)

This script sets up Kaggle access, downloads the dataset, prepares image generators,
builds a CNN model, trains it, evaluates it, and finally allows testing on custom images.
"""

# --- Kaggle Setup ---
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download the dataset from Kaggle
!kaggle datasets download -d sumn2u/garbage-classification-v2

# Extract the archive into a folder
!unzip -q garbage-classification-v2.zip -d /content/dataset/

!ls /content/dataset

# --- Library Imports ---
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

print("Using TensorFlow:", tf.__version__)

# Directory paths for training and testing
train_dir = '/content/dataset/garbage-dataset/train'
test_dir  = '/content/dataset/garbage-dataset/test'

# --- Preparing Image Data ---
data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_loader = data_gen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    batch_size=32,
    subset='training'
)

val_loader = data_gen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    batch_size=32,
    subset='validation'
)

# --- CNN Architecture ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(train_loader.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Train the Model ---
history = model.fit(
    train_loader,
    validation_data=val_loader,
    epochs=10
)

# Save your trained model
model.save('waste_classifier_model.h5')
print("Model has been saved successfully!")

# --- Plot Training History ---
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Trend')
plt.legend()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Trend')
plt.legend()
plt.show()

# --- Test on a Single Image ---
import numpy as np
from tensorflow.keras.preprocessing import image

sample_path = '/content/dataset/garbage-dataset/plastic/plastic1.jpg'  # adjust the filename if needed

img = image.load_img(sample_path, target_size=(128,128))
img_arr = image.img_to_array(img) / 255.0
img_arr = np.expand_dims(img_arr, axis=0)

pred = model.predict(img_arr)
pred_class = list(train_loader.class_indices.keys())[np.argmax(pred)]

plt.imshow(image.load_img(sample_path))
plt.axis('off')
plt.title(f'Predicted class: {pred_class}')
plt.show()

print("Result:", pred_class)
print("Image tensor shape:", img_arr.shape)

# Download model
from google.colab import files
files.download('waste_classifier_model.h5')

# Recreate the same architecture for later loading (optional)
model_rebuild = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(10, activation='softmax')
])

model_rebuild.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
