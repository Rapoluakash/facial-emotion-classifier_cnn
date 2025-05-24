# Facial Emotion Classifier (CNN)

A Convolutional Neural Network (CNN)-based binary image classifier that detects facial emotion: **Happy** or **Not Happy**. Built using TensorFlow and Keras, this project classifies images based on facial expressions.

---

## 🚀 Features

- Image classification using a custom CNN model
- Binary emotion detection: "Happy" vs "Not Happy"
- Uses `ImageDataGenerator` for efficient training and validation
- Compatible with `.jpg`, `.png` images

---

## 🧠 Model Architecture

- 3 convolutional layers with ReLU activation and max-pooling
- Flatten layer followed by a fully connected dense layer
- Binary classification output with sigmoid activation
