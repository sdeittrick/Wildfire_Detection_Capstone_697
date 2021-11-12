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
import os

def predict(img):
    dir = '/'.join(os.getcwd().split('/')[:-1]) + '/wildfire_prediction_pipeline'
    model = tf.keras.models.load_model('{}/model.h5'.format(dir))
    
    classes = {1: 'Fire Detected', 0: 'No Fire Detected'}

    width = model.input_shape[2]
    height = model.input_shape[1]

    img = Image.open(img)
    img = img.resize((width, height))
    img = np.array(img)/255.00
    img_array = tf.expand_dims(img, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    prediction = classes[np.argmax(score)]

    return prediction

if __name__ == '__main__':
    import argparse

    dir = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='prediction image')
    args = parser.parse_args()

    prediction = predict(args.image)

    print(prediction)
