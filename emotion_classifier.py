import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# === Set Paths ===
train_dir = r"C:\Users\rapol\OneDrive\Documents\Desktop\cnn\training"
val_dir = r"C:\Users\rapol\OneDrive\Documents\Desktop\cnn\validation"
test_dir = r"C:\Users\rapol\OneDrive\Documents\Desktop\cnn\teasting"

# === Image Generators ===
train = ImageDataGenerator(rescale=1./255)
validation = ImageDataGenerator(rescale=1./255)

train_dataset = train.flow_from_directory(
    train_dir,
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary'
)

validation_dataset = validation.flow_from_directory(
    val_dir,
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary'
)

# === Build Model ===
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(200, 200, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    metrics=['accuracy']
)

# === Train Model ===
model.fit(train_dataset, epochs=9)

# === Save Model (optional) ===
model.save("emotion_classifier_model.h5")

# === Predict on Test Images ===
def predict_emotions(directory):
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = load_img(img_path, target_size=(200, 200))
        plt.imshow(img)
        plt.axis('off')
        plt.title(filename)
        plt.show()

        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize

        prediction = model.predict(x)

        if prediction[0][0] < 0.5:
            print(f"{filename}: I am happy")
        else:
            print(f"{filename}: I am not happy")

# === Run Predictions ===
predict_emotions(test_dir)
