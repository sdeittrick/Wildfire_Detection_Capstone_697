#import required libraries
import os
import pandas as pd
import numpy as np
import pickle

#image processing libraries
import PIL
from PIL import Image
import matplotlib.pyplot as plt

def resize_image(img):
    """resize images prior to utilizing in trianing model"""    
    width, height = img.size
    ratio = width/height
    new_height = 100
    new_width = int(new_height*ratio)
    img = img.resize((new_width, new_height))
    return img

def get_image_stats(image_arrays):
    """get image stats for loading in images"""
    widths = [array.shape[0] for array in image_arrays]
    lengths = [array.shape[1] for array in image_arrays]
    width_mean = np.mean(widths)
    length_mean = np.mean(lengths)
    width_max = np.max(widths)
    length_max = np.max(lengths)
    return length_mean, length_max, width_max, width_mean

def load_images(img_dir, type, height):
    ''' Type as int 0 no_fire 1 fire
        desired height of image as int 
        img_dir as str folder name where images stored'''

    #get list of images 
    dir = os.getcwd()
    images = os.listdir('{}/input_images/{}'.format(dir, img_dir))
    image_file = [file for file in images if '.DS_Store' not in file]
    image_arrays = [np.asarray(resize_image(Image.open('{}/input_images/{}/{}'.format(dir,img_dir, image)))) for image in image_file]
    
    #get_labels
    label_arrays = []
    for array in image_arrays:
        label_arrays.append([int(type)])
    image_labels = label_arrays

    data = list(zip(image_arrays, image_labels))

    return data

if __name__ == '__main__':
    import argparse

    dir = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('input_directory', help='image_directory')
    parser.add_argument('image_type', help='0 - Nonfire, 1 - fire')
    parser.add_argument('height', help='height of image')
    parser.add_argument('output_file', help='data_file_name')
    args = parser.parse_args()

    data = load_images(args.input_directory, args.image_type, args.height)

    with open('{}/artifacts/loaded_pickles/{}'.format(dir, args.output_file), 'wb+') as out:
        pickle.dump(data, out)