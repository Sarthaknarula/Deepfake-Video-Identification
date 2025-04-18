import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import load_model

!kaggle datasets download -d manjilkarki/deepfake-and-real-images

!unzip deepfake-and-real-images.zip

# Define dataset paths (Kaggle-specific)
train_dir = '/content/Dataset/Train'
validation_dir = '/content/Dataset/Validation'
test_dir = '/content/Dataset/Test'



# Load and preprocess the data
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_flow = train_datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')
validation_flow = validation_datagen.flow_from_directory(validation_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')
test_flow = test_datagen.flow_from_directory(test_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')




# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))



# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


from tensorflow.keras.callbacks import ModelCheckpoint

# Define the checkpoint callback
checkpoint = ModelCheckpoint(
    filepath='best_model.keras',      # Path where the model will be saved
    monitor='val_loss',            # Metric to monitor
    save_best_only=True,           # Save only the best model
    save_weights_only=False,       # Save the entire model (architecture + weights)
    mode='min',                    # Save when the monitored quantity is minimized
    verbose=1                      # Print updates during the training
)

# Train the model with the checkpoint callback
history = model.fit(
    train_flow,
    epochs=10,
    validation_data=validation_flow,
    callbacks=[checkpoint]         # Include the checkpoint callback in training
)



# Train the model
# history = model.fit(train_flow, epochs=10, validation_data=validation_flow)


model.save('deepfake_new.keras')

# Plot training & validation loss values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.show()


# Evaluate the model
loss, accuracy = model.evaluate(test_flow)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
