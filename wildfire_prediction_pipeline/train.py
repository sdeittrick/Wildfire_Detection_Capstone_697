import os
import pandas as pd
import numpy as np
import pickle

import PIL
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorboard.plugins.hparams import api as hp
from keras import backend as K

from sklearn.metrics import confusion_matrix
from sklearn import metrics

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def areaUnderCurve(y_true, y_pred):
    areaUnder = metrics.auc(y_true, y_pred)
    return areaUnder


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

def augment_layers(input_shape):

    augLayers =[
    layers.RandomFlip("vertical",input_shape=input_shape),
    layers.RandomRotation(0.1),
    # layers.RandomZoom(0.1),
    #other options we can run with.. black and white, saturation, brightness, etc...
    # layers.RandomContrast(1.0, seed=100),
    ]

    return keras.Sequential(augLayers)

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


def train_test_model(hparams,epochs,input_shape,train_images,train_labels,test_images, test_labels,
                    HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, predictions,
                    params, losses, accuracies, f1_scores, precisions, recalls, cms, aucs,
                    units, dropouts, optimizers, histories, augmentModel=True):
    
    if augmentModel:
        data_augmentation = augment_layers(input_shape)
    else:
        data_augmentation = ''
    
    model = tf.keras.models.Sequential([
    # data_augmentation,
    # tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    # layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    # layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    # layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation='relu'),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    # layers.Dense(1, activation='sigmoid')
    ])

    METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'), 
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),
    ]

    model.compile(
      optimizer=hparams[HP_OPTIMIZER],
    #  loss='sparse_categorical_crossentropy',
      loss=tf.keras.losses.BinaryCrossentropy(),
       metrics=METRICS
    #  metrics=['accuracy',f1_m,precision_m, recall_m]#,areaUnderCurve],
    )
    

    model.fit(train_images, train_labels, epochs=epochs)#validation_data=(test_images, test_labels),
    model.save("artifacts/model_all.h5")
    # history = model.fit(train_images, train_labels, epochs=epochs)
    histories.append(model.history)
    loss, tp, fp, tn, fn, accuracy, precision, recall, auc, prc= model.evaluate(test_images, test_labels)
    
    losses.append(loss)
    accuracies.append(accuracy)
    #f1_scores.append(f1_score)
    precisions.append(precision)
    recalls.append(recall)
    aucs.append(auc)
    
    # prediction = model.predict(x=test_images, steps=len(test_images), verbose=0)
    # predictions.append(prediction)

    predictions_probs = model.predict(test_images, verbose = 1)

    scores = []
    for prediction in predictions_probs:
        if prediction >= .5:
            scores.append(1)
        else:
            scores.append(0)
    predictions.append(scores)

    cm = confusion_matrix(y_true=test_labels, y_pred=scores)#y_pred=np.argmax(prediction, axis=-1)
    cms.append(cm)

    #calculate AUC of model
    # auc = metrics.roc_auc_score(test_labels, y_pred_proba)
    # auc, update_op_auc = tf.metrics.auc(y_true, y_scores, name = 'AUC')
    # auc_value, auc_op = tf.metrics.auc(test_labels, predictions[:, 1])
    # print(auc)
    aucs.append(0)
    
    # return loss, accuracy, f1_score, precision, recall#, areaUnder
    return model

def run(run_dir, hparams, epochs, input_shape,train_images,train_labels,test_images, test_labels,
        HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, predictions,
        params, losses, accuracies, f1_scores, precisions, recalls, cms, aucs,
        units, dropouts, optimizers, histories, augmentModel=True):

    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial

        model = train_test_model(hparams,epochs,input_shape,train_images,train_labels,test_images, test_labels,
                                HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, predictions,
                                params, losses, accuracies, f1_scores, precisions, recalls, cms, aucs,
                                units, dropouts, optimizers, histories, augmentModel)
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
    