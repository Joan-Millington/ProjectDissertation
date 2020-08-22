# From Connor Shorten's 'Image Classification Keras Tutorial: Kaggle Dog 
# Breed Challenge'. Adapted to process the UCI Machine Learning Repository's 
# 'Rice Leaf Diseases Data Set' (Shah et al.) and adapted to run on a 
# Microsoft Azure Machine Learning virtual machine (GPU).
#
# Joan Millington, Project Dissertation Aug 2020.

from PIL import Image
import numpy as np
import os
import random 
import matplotlib.pyplot as plt 

# This script trains a simple CNN model to predict one of three
# rice leaf diseases and evaluates that models' accuracy

# Azure 
try:    
    from azureml.core.run import Run
    run = Run.get_submitted_run()
except:
    run = None

# train_test_dataset_all contains 120 JPGs, each image showing one of three diseases categories.
myPathInput = r".\train_test_dataset_all"

# Verify that the path exists
if not os.path.exists(myPathInput):
    raise IOError("Input Folder does not exist: {}".format(myPathInput))

# Verify that the folder contains JPGs
def checkFilesJPG(DIR):
    for root, dirs, files in os.walk(myPathInput, topdown=False):
        for name in files:
            if not name.endswith('JPG') and not ('jpg'):
                raise IOError("Input Folder must only contain jpg files. Please remove file: {}".format(name))  

checkFilesJPG(myPathInput)

# Want to know how we should format the height x width image data dimensions
# for inputting to a keras model, display the results.
def get_size_statistics(DIR):    
    heights = []
    widths = []
    for img in os.listdir(DIR): 
        path = os.path.join(DIR, img)
        data = np.array(Image.open(path)) #PIL Image library
        heights.append(data.shape[0])
        widths.append(data.shape[1])
    avg_height = sum(heights) / len(heights)
    avg_width = sum(widths) / len(widths)  
    print("Average Height: " + str(avg_height))
    print("Max Height: " + str(max(heights)))
    print("Min Height: " + str(min(heights)))
    print("Average Width: " + str(avg_width))
    print("Max Width: " + str(max(widths)))
    print("Min Width: " + str(min(widths)))

get_size_statistics(myPathInput)

# Label images by one of three diseases: 
# bacterial leaf blight (blb), brown spot (bs) and leaf smut (ls)
def label_img(name):     
    word_label = name.split('_')[0]
    if word_label == 'blb' : return np.array([1, 0, 0])
    elif word_label == 'bs' : return np.array([0, 1, 0])
    elif word_label == 'ls' : return np.array([0, 0, 1])

# Reduced the image pixels to 500 (instead of 300) because differences are slight.
IMG_SIZE = 500

def load_training_data():
    train_data = []
    for img in os.listdir(myPathInput):
        label = label_img(img)
        path = os.path.join(myPathInput, img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        train_data.append([np.array(img), label])    

        # Data Augmentation - Horizontal Flipping
        flip_img = Image.open(path)
        flip_img = flip_img.convert('L')
        flip_img = flip_img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        flip_img = np.array(flip_img)
        flip_img = np.fliplr(flip_img)
        train_data.append([flip_img, label])  
    random.shuffle(train_data)
    return train_data

training_data = load_training_data()
# Plot is hashed out because it cannot be displayed from a script running in the 
# cloud, when instigated from an IDE on a laptop (a networked user interface to 
# the cloud would be required). 
# plt.imshow(training_data[43][0], cmap = 'gist_gray')

trainImages = np.array([i[0] for i in training_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
trainLabels = np.array([i[1] for i in training_data])

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization

def simple_cnn_model_and_evaluation(data):
    model = keras.Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', 
    input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))    
    #model.add(Dropout(0.3))
    model.add(Dense(3, activation = 'softmax'))     # Changed from 2 to 3 for 3 disease categories

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    model.fit(trainImages, trainLabels, batch_size = 30, epochs = 7, verbose = 1)

    loss, acc = model.evaluate(trainImages, trainLabels, verbose = 0)
    print('Accuracy: ', acc * 100)

# The load_test_data function has been removed because the rice leaf dataset only 
# contains 120 images and is quick to run. (This script was designed for 10,222 dog images)

simple_cnn_model_and_evaluation(load_training_data())

# Azure
if run is not None:
    run.log('Images resized to ', IMG_SIZE)
