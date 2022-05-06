import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.pyplot import show
import numpy as np
from skimage.feature import hog
import joblib

import warnings
warnings.filterwarnings("ignore")
# the warning comes from the fact that the model is initially trained on a dataset while prediction is made on single value. safe to ignore

# Feature extraction functions

def remove_scrollbar(image, bar_width = 20):
    # gets rid of scroll bar so that model wont learn from that and returns the image
    y,x,_ = image.shape
    img = image[0:y,0:x-bar_width]
    return img


def extract_features(resized_image, orientations=10, pixels_per_cell=(6,6), cells_per_block=(1, 1)):
    # applied on resized image
    fd, hog_image = hog(resized_image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True, channel_axis=-1)
    return fd, hog_image


def resize_image(image, new_size = (256, 128)): # size to resize to after future extraction
    return cv.resize(image, new_size)


def resize_and_get_feature_vector(image):
    # returns 1d feature vector to be appended to df
    resized_image = resize_image(image)
    fd, hog_image = extract_features(resized_image)

    return fd

# Dataset preperation functions

def get_fieldnames(n_features = 8820): # with image of size (256, 128) 
    fieldnames = ['file_name', 'label']
    feature_values =  ['x_'+str(i) for i in range(n_features)]
    fieldnames += feature_values

    return fieldnames

def get_row(file_name, label, fd):
    new_row = {'file_name': file_name, 'label': label}
    for i,px_val in enumerate(fd.flatten()):
        new_row['x_'+str(i)] = px_val

    return new_row

model_path = {1: "models/svm_best_pipeline.joblib", 2: 'models/sgd_best_pipeline.joblib'}


# Inference
def load_model(model_nr):
    classifier = joblib.load(model_path[model_nr])
    
    return classifier

def infer_image(image_path, model_nr):
    classifier = load_model(model_nr)
    image = cv.imread(image_path)

    # feature extraction
    fd = resize_and_get_feature_vector(image)
    # 2d array is expected
    prediction = classifier.predict(fd.reshape(1,-1))

    return prediction[0]
