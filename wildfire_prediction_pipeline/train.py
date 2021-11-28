import os
import pandas as pd
import numpy as np
import pickle

import PIL
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential


def test_train_split(data):
    # randomize the images
    import random 
    random.seed(42)
    random.shuffle(data)

    #split into train, test, dev
    data_len = len(data)
    train, dev, test = np.split(data, [int(data_len*.8),int(data_len*.9)])
    train_images, train_labels = zip(*train)
    dev_images, dev_labels = zip(*dev)
    test_images, test_labels = zip(*test)
    
    return train_images, train_labels, dev_images, dev_labels, test_images, test_labels



def cnn_model(data):
    
    train_images, train_labels, dev_images, dev_labels, test_images, test_labels = test_train_split(data)
    
    train_images, dev_images, test_images = np.array(train_images) / 255.0, \
                                            np.array(dev_images) / 255.0, \
                                            np.array(test_images) / 255.0



    train_images = tf.convert_to_tensor(train_images, dtype=tf.float32)
    train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
    test_images = tf.convert_to_tensor(test_images, dtype=tf.float32)
    test_labels = tf.convert_to_tensor(test_labels, dtype=tf.float32)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 178, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    epochs=10

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=epochs, 
                        validation_data=(test_images, test_labels))
    return model

if __name__ == '__main__':
    import argparse

    dir = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('processed_images', help='image_file')
    parser.add_argument('output_file', help='data_file_name')
    args = parser.parse_args()

    with open('{}/artifacts/{}'.format(dir, args.processed_images), 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    model = cnn_model(data)
    model = model.save("model.h5")
    