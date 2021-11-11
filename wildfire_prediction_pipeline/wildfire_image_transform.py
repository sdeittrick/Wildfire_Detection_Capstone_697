#import required libraries
import os
import pandas as pd
import numpy as np
import pickle

#image processing libraries
import PIL
from PIL import Image

def get_image_stats(image_arrays):
    widths = [array.shape[0] for array in image_arrays]
    lengths = [array.shape[1] for array in image_arrays]
    width_mean = np.mean(widths)
    length_mean = np.mean(lengths)
    width_max = np.max(widths)
    length_max = np.max(lengths)
    return length_mean, length_max, width_max, width_mean

def pad_images(images, width_max, length_max): 
    """pad images prior to utilizing in trianing model""" 
    padded_images = [] 
    c = 0 
    for image in images: 
        wpad1 = (width_max - image.shape[0])/2
        lpad1 = (length_max - image.shape[1])/2

        if lpad1%1 > 0: 
            lpad1 = int(lpad1)
            lpad2 = int(lpad1)+1
        else: 
            lpad1 = int(lpad1)
            lpad2 = int(lpad1)

        if wpad1%1 > 0: 
            wpad1 = int(wpad1)
            wpad2 = int(wpad1)+1
        else: 
            wpad1 = int(wpad1)
            wpad2 = int(wpad1)
        try:
            padded_images.append(np.pad(image, pad_width=[(wpad1 , wpad2),(lpad1, lpad2),(0, 0)], mode='constant'))
        except: 
            print("at {}".format(c))
        c = c + 1
    return padded_images

def crop_images(images, width_mean, length_mean):
    """Crop images"""    
    train_padded_c = []
    for image in images: 
        
        left = int((image.shape[0] - int(width_mean))/2)
        top = int((image.shape[1] - int(length_mean))/2)
        right = int((image.shape[0] + int(width_mean))/2)
        bottom = int((image.shape[1] + int(length_mean))/2)

        train_padded_c.append(image[left:right, top:bottom])
    return train_padded_c

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('pickle_dir', help='directory of loaded pickles')
    parser.add_argument('output_file', help='data_file_name')
    args = parser.parse_args()

    dir = os.getcwd()
    
    data = []
    for pkl in os.listdir('{}/artifacts/{}'.format(dir, args.pickle_dir)):
        with open('{}/artifacts/{}/{}'.format(dir, args.pickle_dir, pkl), 'rb') as pickle_file:
            data.extend(pickle.load(pickle_file))

    images, labels = zip(*data)
    length_mean, length_max, width_max, width_mean = get_image_stats(images)
    images_final = crop_images(pad_images(images, width_max, length_max), width_mean, length_mean)

    images_final = [np.array(Image.fromarray(image).convert('RGB')) for image in images_final]
    data = list(zip(images_final, labels))

    with open('{}/artifacts/{}'.format(dir,args.output_file), 'wb+') as out:
        pickle.dump(data, out)