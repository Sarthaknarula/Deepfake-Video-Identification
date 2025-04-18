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
# model = tf.keras.models.load_model('/content/drive/MyDrive/baseline_model_fold_1.keras')

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize) ) + 1
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

def load_images(files):
    images=[]
    for i in files:
        a = plotstft(i)
#         a = Image.open(i)
        a = np.asarray(a)
        a = cv2.resize(a, (360,360))
        a = np.reshape(a, (360,360,1))
#         a = a/255.0
        images.append(a)
    return np.array(images)

# BATCH_SIZE = 49
def DataGen(data,label,BATCH_SIZE):
    while True:
        for i in range(len(data)):
#             rand = np.random.randint(0,len(data)-BATCH_SIZE)
            images = load_images(data[i:i+BATCH_SIZE])
            labels = label[i:i+BATCH_SIZE]
            yield np.array(images),np.array(labels)


def predict_audio(file_path, toggle = -1):
    if (toggle == -1):
      print("are the videos fake or real (1 : fake, 0 : real)")
      toggle = float(input())
    ans = 0;

    from typing import final
    test_df = pd.DataFrame(columns=['spectrogram','label'])
    test_df['spectrogram'] = file_path
    test_df['label'] = toggle
    # print(test_df)
    # print(len(test_df))
    test_dataset = DataGen(test_df['spectrogram'],test_df['label'], test_df.size-1)
    # a = test_dataset['spectrogram']
    # b = test_dataset['label']
    a, b = next(test_dataset)

    # from PIL import Image
    # import numpy as np

    # # Example: create or load a numpy array (grayscale or RGB)
    # array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)  # Random RGB image

    # # Convert the numpy array to a PIL Image object
    # img = Image.fromarray(array)

    # # Save the image to the desired directory
    # img.save('abc.jpg')

    # a.save('abc')

    print(test_df.shape)
    print(a.shape)

    # y_pred = model.predict(a)
    # print(y_pred.shape)
    # y_pred = None
    # for a, b in test_dataset:
    #   y_pred = model.predict(a)
    #   for i in y_pred:
    #     if (toggle == 1 and i >= 0.03):
    #       ans = ans + 1
    #     elif (toggle == 0 and i < 0.03):
    #       ans = ans + 1

    y_pred = model.predict(a)
    for i in y_pred:
      if (toggle == 1 and i >= 0.03):
        ans = ans + 1
      elif (toggle == 0 and i < 0.03):
          ans = ans + 1
      print(i);
    print(ans)
    print(len(y_pred))
    print("accuracy % = ",100*ans/len(y_pred))

# !kaggle datasets download -d mohammedabdeldayem/the-fake-or-real-dataset
# !unzip the-fake-or-real-dataset.zip
!kaggle datasets download -d f0rtaza/fake-audio
!unzip fake-audio.zip
# !kaggle datasets download -d abdallamohamed312/in-the-wild-dataset
# !unzip in-the-wild-dataset.zip

# !pip install --upgrade tensorflow==2.13

# import tensorflow as tf
# # import tensorflow.keras as keras
# print(tf.__version__)
# print(keras.__version__)

# model = tf.keras.models.load_model('/content/baseline_model_final3 (1).keras')

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

# Print model summary
model.summary()


# model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
model.load_weights('/content/baseline_model_final3 (1).keras')

# !kaggle datasets download -d mohammedabdeldayem/the-fake-or-real-dataset
# !unzip the-fake-or-real-dataset.zip

file_path = glob.glob("/content/fake_audio/Original_audios/*.wav")
toggle = 0
predict_audio(file_path, 0)