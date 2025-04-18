!pip install pandas==1.3.3

import os
import glob
import json
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

import sklearn
from sklearn.model_selection import train_test_split

import librosa as lb
import wave
import pylab
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

# def get_wav_info(wav_file):
#     wav = wave.open(wav_file, 'r')
#     frames = wav.readframes(-1)
#     sound_info = pylab.fromstring(frames, 'int16')
#     frame_rate = wav.getframerate()
#     wav.close()
#     return sound_info, frame_rate

# def graph_spectrogram(wav_file):
#     sound_info, frame_rate = get_wav_info(wav_file)
#     pylab.figure(num=None, figsize=(19, 12))
#     pylab.subplot(111)
#     pylab.title('spectrogram of %r' % wav_file)
#     pylab.specgram(sound_info, Fs=frame_rate)
#     pylab.savefig('spectrogram.png')


""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

""" scale frequency axis logarithmically """
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    epsilon = 1e-10
    # ims = 20.*np.log10(np.abs(sshow)/(10e-6+epsilon)) # amplitude to decibel

    ims = np.abs(sshow)
    ims = np.where(ims < epsilon, epsilon, ims)  # Replace values less than epsilon with epsilon
    ims = 20. * np.log10(ims / (10e-6))  # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    return ims

# !kaggle datasets download -d f0rtaza/fake-audio
# !unzip fake-audio.zip
# !kaggle datasets download -d abdallamohamed312/in-the-wild-dataset
# !unzip in-the-wild-dataset.zip
!kaggle datasets download -d mohammedabdeldayem/the-fake-or-real-dataset
!unzip the-fake-or-real-dataset.zip

# !unzip /kaggle/input/in-the-wild-dataset/download -d /kaggle/working
# !ls /content/download

# fake_audio = glob.glob("/content/fake_audio/Fake_audios/*.wav")
# original_audio = glob.glob("/content/fake_audio/Original_audios/*.wav")

# fake_audio = glob.glob("/content/for-norm/for-norm/training/fake/*.wav")
# original_audio = glob.glob("/content/for-norm/for-norm/training/real/*.wav")
# fake_audio.extend(glob.glob("/content/for-norm/for-norm/validation/fake/*.wav"))
# original_audio.extend(glob.glob("/content/for-norm/for-norm/validation/real/*.wav"))
# fake_audio.extend(glob.glob("/content/for-original/for-original/training/fake/*.wav"))
# original_audio.extend(glob.glob("/content/for-original/for-original/training/real/*.wav"))
fake_audio = (glob.glob("/content/for-original/for-original/validation/fake/*.wav"))
original_audio = (glob.glob("/content/for-original/for-original/validation/real/*.wav"))
audio_file = glob.glob("/content/download")

# print(len(audio_file))
print(len(fake_audio),len(original_audio))


# fake_audio_arr = []
# original_audio_arr = []
# for i in range(len(fake_audio)):
#     img = plotstft(fake_audio[i])
#     fake_audio_arr.append(img)

# for i in range(len(original_audio)):
#     img = plotstft(original_audio[i])
#     original_audio_arr.append(img)

# fake_audio_arr = np.array(fake_audio_arr)
# original_audio_arr = np.array(original_audio_arr)
# print(fake_audio_arr.shape, original_audio_arr.shape)

# !pip install pandas==1.3.3

# import pandas as pd
# train_df = pd.DataFrame(columns=['spectrogram','label'])
# print(type(train_df))  # This should output <class 'pandas.core.frame.DataFrame'>
# # train_df.conc
# print(pd.__version__)


train_df = pd.DataFrame(columns=['spectrogram','label'])
for i in range(36000):
    train_df = train_df.append({'spectrogram':fake_audio[i],'label':1.0},ignore_index=True)
    train_df = train_df.append({'spectrogram':original_audio[i],'label':0.0},ignore_index=True)

train_df.head()

X= train_df['spectrogram']
y= train_df['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=23)

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


# def load_images(files):
#     images = []
#     valid_files = []

#     for file in files:
#         if not os.path.exists(file):
#             print(f"File not found: {file}")
#             continue

#         try:
#             a = plotstft(file)  # Assuming plotstft returns a numpy array
#             if a is None or a.size == 0:
#                 print(f"Warning: Empty or invalid data from file {file}. Skipping.")
#                 continue  # Skip this image

#             a = np.asarray(a)

#             # Ensure the image has the correct shape
#             if a.shape[0] != 360 or a.shape[1] != 360:
#                 print(f"Warning: Image at {file} has unexpected shape {a.shape}. Resizing...")
#                 try:
#                     a = cv2.resize(a, (360, 360))
#                 except cv2.error as e:
#                     print(f"Error resizing image at {file}: {e}. Skipping.")
#                     continue  # Skip this image

#             a = np.reshape(a, (360, 360, 1))  # Ensure the correct shape (360, 360, 1)
#             images.append(a)
#             valid_files.append(file)  # Keep track of valid files

#         except Exception as e:
#             print(f"Error processing file {file}: {e}. Skipping.")

#     print(f"Loaded {len(images)} valid images out of {len(files)} total files.")
#     return np.array(images), valid_files


def load_images(files):
    images = []
    for file in files:
        if not os.path.exists(file):
            print(f"File not found: {file}")
            continue

        a = plotstft(file)  # Assuming plotstft returns a numpy array
        if a is None or a.size == 0:
            print(f"Warning: Empty or invalid image at {file}")
            continue  # Skip this image

        a = np.asarray(a)

        # Ensure the image has the correct shape
        if a.shape[0] != 360 or a.shape[1] != 360:
            # print(f"Warning: Image at {file} has unexpected shape {a.shape}. Resizing...")
            try:
                a = cv2.resize(a, (360, 360))
            except cv2.error as e:
                print(f"Error resizing image at {file}: {e}")
                continue  # Skip this image
            except Exception as e:
                print(f"Empty or invalid image")
                continue  # Skip this image

        a = np.reshape(a, (360, 360, 1))  # Ensure the correct shape (360, 360, 1)
        images.append(a)

    return np.array(images)


# def load_images(files):
#     images=[]
#     for i in files:
#         a = plotstft(i)
# #         a = Image.open(i)
#         a = np.asarray(a)
#         a = cv2.resize(a, (360,360))
#         a = np.reshape(a, (360,360,1))
# #         a = a/255.0
#         images.append(a)
#     return np.array(images)

# Batch_size = 16

# def DataGen(data, label, BATCH_SIZE=Batch_size):
#     data_length = len(data)
#     while True:
#         for start in range(0, data_length, BATCH_SIZE):
#             end = min(start + BATCH_SIZE, data_length)
#             images = load_images(data[start:end])
#             labels = label[start:end]
#             yield np.array(images), np.array(labels)

# train_dataset = DataGen(X_train, y_train, BATCH_SIZE=Batch_size)
# valid_dataset = DataGen(X_test, y_test, BATCH_SIZE=Batch_size)

# a, b = next(train_dataset)
# c, d = next(valid_dataset)

# print(a.shape, b.shape, c.shape, d.shape)


Batch_size = 16
def DataGen(data,label,BATCH_SIZE = Batch_size):
    while True:
        for i in range(len(data)):
#             rand = np.random.randint(0,len(data)-BATCH_SIZE)
            images = load_images(data[i:i+BATCH_SIZE])
            labels = label[i:i+BATCH_SIZE]
            yield np.array(images),np.array(labels)

train_dataset = DataGen(X_train,y_train, 48)
valid_dataset = DataGen(X_test,y_test, 16)

a,b = next(train_dataset)
c,d = next(valid_dataset)
print(a.shape,b.shape,c.shape,d.shape)
print(X_train.size, X_test.size)

def sample_plot(image,label):
    fig, axs = plt.subplots(5, 1, figsize=(10,10))
    fig.tight_layout()
    for i in range(5):

        axs[i].imshow(image[i])
        axs[i].set_title('Real' if label[i]==0.0 else 'Fake')

#         axs[i,1].imshow(masks[i])
#         axs[i,1].set_title('Mask')

    plt.show()
sample_plot(a,b)

# inputs = tf.keras.layers.Input((360, 360, 1))

# # Data augmentation
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip("horizontal"),
#     tf.keras.layers.RandomTranslation(0.1, 0.1),
# ])(inputs)

# # Load the EfficientNetB1 model
# base_model = tf.keras.applications.EfficientNetB1(include_top=False, weights='imagenet', pooling='max')
# base_model.trainable = True

# # Pass the augmented data through the base model
# x = base_model(data_augmentation, training=True)

# # Apply dropout (corrected)
# x = tf.keras.layers.Dropout(0.2)(x)

# # Add dense layers
# x = tf.keras.layers.Dense(16, activation='relu')(x)

# # Output layer
# outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# # Build and compile the model
# model = tf.keras.Model(inputs, outputs)
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#               loss='binary_crossentropy',
#               metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5), tf.keras.metrics.AUC()])


inputs = tf.keras.layers.Input((360,360,1))
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
#   layers.RandomRotation(0.2),
  layers.RandomTranslation(0.1,0.1),
#   layers.RandomContrast(0.1)
])(inputs)

base_model = tf.keras.applications.efficientnet.EfficientNetB1(include_top=False, weights='imagenet',pooling='max')
base_model.trainable=True
x = base_model(data_augmentation,training=True)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5),tf.keras.metrics.AUC()])

model.summary()



# Batch_size = 16
# EPOCHS = 5
# # checkpoint = tf.keras.callbacks.ModelCheckpoint('baseline_model.keras', monitor='val_auc', verbose=1, save_best_only=True, mode='max')
# checkpoint = tf.keras.callbacks.ModelCheckpoint('baseline_model_new.keras', monitor='val_auc', verbose=1, save_best_only=True, mode='max')
# lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_binary_accuracy',patience=3,verbose=1,factor=0.5,min_lr=1e-5)


# # Update train_dataset and valid_dataset generators
# train_dataset = DataGen(X_train, y_train, 48)
# valid_dataset = DataGen(X_test, y_test, 16)

# # Calculate correct steps per epoch based on the batch size
# TRAIN_STEPS = len(X_train) // Batch_size
# VAL_STEPS = len(X_test) // Batch_size

# history = model.fit(
#     train_dataset,
#     steps_per_epoch=TRAIN_STEPS,
#     validation_data=valid_dataset,
#     validation_steps=VAL_STEPS,
#     epochs=EPOCHS,
#     callbacks=[checkpoint, lr_reducer]
# )


for batch_images, batch_labels in train_dataset:
    print(batch_images.shape, batch_labels.shape)
    break

# Set up the ModelCheckpoint callback to save the model after every epoch
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'baseline_model_final.keras',  # Path to save the model
    monitor='val_loss',  # Can be any metric, won't affect saving due to save_best_only=False
    verbose=1,
    save_best_only=True,  # Save the model at the end of every epoch
    save_weights_only=False,  # Save the entire model
    mode='min',  # Mode is set to auto, as save_best_only is False
    save_freq='epoch'  # Save every epoch
)

# Reduce learning rate on plateau callback
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_binary_accuracy',
    patience=3,
    verbose=1,
    factor=0.5,
    min_lr=1e-5
)

# Define training parameters
TRAIN_STEPS = 48
VAL_STEPS = 16
EPOCHS = 1125

# Train the model
# history = model.fit(
#     train_dataset,
#     steps_per_epoch=TRAIN_STEPS,
#     validation_data=valid_dataset,
#     validation_steps=VAL_STEPS,
#     epochs=EPOCHS,
    #  callbacks=[checkpoint, lr_reducer]
# )


# in the model
history = model.fit(
    train_dataset,
    steps_per_epoch=TRAIN_STEPS,
    validation_data=valid_dataset,
    validation_steps=VAL_STEPS,
    epochs=EPOCHS,
    callbacks=[checkpoint, lr_reducer]
)

import matplotlib.pyplot as plt
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

model.save('baseline_model.keras')