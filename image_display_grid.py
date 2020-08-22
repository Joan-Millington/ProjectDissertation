# Based on the image display code within Dmitry Soshnikov's 'AzureMlStarter'
# (github), blog: https://soshnikov.com/azure/best-way-to-start-with-azureml/ 
# The dataset is 'Rice Leaf Diseases Data Set' (Shah et al.) published on 
# UCI's Machine Learning Repository. 
#
# Joan Millington, Project Dissertation Aug 2020.

from PIL import Image
import numpy as np
import sys
import os

# Runs in VS Code on local device, with dissenv Interpreter (environment) 
# for image display window

# Prepare the dimensions of the image display grid
import matplotlib.pyplot as plt 
columns = 10         
rows = 12
numJpgs = columns * rows
fig,ax=plt.subplots(1,numJpgs)   

# Folder with JPGs
myDir = r".\train_test_dataset_all"

# Check that the above folder exists and that it contains JPGs
def createFileList(DIR):
    fileList = []
    for root, dirs, files in os.walk(DIR, topdown=False):
        for name in files:
            if name.endswith('JPG') or ('jpg'):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

myFileList = createFileList(myDir)

# Add next image to the columns x rows display grid 
# (provided the count does not exceed the grid dimensions)
def preparePlot(displayImage, count):
    if count <= numJpgs:
        displayImage.thumbnail((28,28))
        fig.add_subplot(rows, columns, count)
        plt.imshow(displayImage)

count = 1

# Prepare each image for display
for file in myFileList:
    leafImage = Image.open(file)
    width, height = leafImage.size
    format = leafImage.format
    mode = leafImage.mode
    preparePlot(leafImage, count)
    count += 1

# Hide axis plot lines, ticks and labels
axlineoff_fun = np.vectorize(lambda ax:ax.axis('off'))
axlineoff_fun(ax)

# Open plot window with 120 rice leaf disease images displayed
plt.show()
