import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import load_model

import cv2
import os

def video_to_frames(video_path, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Capture the video from the file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    frame_num = 0

    while True:
        # Read one frame from the video
        ret, frame = cap.read()

        # If the frame was read successfully, save it
        if ret:
            frame_filename = os.path.join(output_dir, f"frame_{frame_num:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_num += 1
        else:
            break

    # Release the video capture object
    cap.release()
    print("Done! Video has been split into frames.")

# Example usage
video_path = '/content/Real_Video.mp4'
output_dir = '/content/Check_Fake2/Accused'
video_to_frames(video_path, output_dir)


test_datagen = ImageDataGenerator(rescale=1./255)

test_dir = '/content/Check_Fake2'
test_flow = test_datagen.flow_from_directory(test_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')
# print(test_flow.head())

import tensorflow as tf
from tensorflow.keras import layers
# Define the Sequential model
model = tf.keras.Sequential([
    # Data Augmentation
    layers.RandomFlip("horizontal", input_shape=(360, 360, 1)),
    layers.RandomTranslation(0.1, 0.1),

    # Pre-trained EfficientNetB1 as feature extractor
    tf.keras.applications.EfficientNetB1(include_top=False, weights='imagenet', pooling='max'),

    # Add dropout and dense layers
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5), tf.keras.metrics.AUC()])

# model.load_weights('/content/baseline_model_final3 (1).keras')

model = tf.keras.models.load_model('/content/deepfake_new.keras')

y_pred = model.predict(test_flow)
score = 0
for i in y_pred:
  if i[1] > 0.5:
    score = score + 1
print("Accuracy of being fake = ", 100*score/len(y_pred))

from re import VERBOSE
import tensorflow as tf

# Add this line before evaluating the model
tf.config.run_functions_eagerly(True)

# Evaluate the model
loss, accuracy = model.evaluate(test_flow, verbose=2)
print(f'Test Accuracy: {accuracy * 100:.2f}%')