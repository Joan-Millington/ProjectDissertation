18th August 2020

Project Dissertation

This code is the early attempt of my project dissertation on 'Artificial Intelligence and Data Commons'.

The aim is to test whether it is possible for someone with little or no python coding experience to combine:

    a publically available algorithm from one source,
    with a publically available database from another source,
    and run it in a cloud environment (without prior link to the algorithm or dataset)
    and deliver a prediction (it doesn't have to be a good prediction, as machine learning skills are not in scope).

To that end:

I have explored these algorithms (so far):

A mildly edited version of train_local.py from Dmitry Soshnikov's 'AzureMlStarter', see blog: https://soshnikov.com/azure/best-way-to-start-with-azureml/. This uses the MNIST dataset from OpenML: https://www.openml.org/d/554 I have edited it to display the handwritten digit images in a 10 x 12 block (as the 'Rice Leaf Diseases Data Set' (see below) has 120 JPGs). However, so far, I have not been able to hide the plot lines so it's cluttered.

image_jpg_to_csv.py is an attempt to convert the 'Rice Leaf Diseases Data Set' JPGs into .csv format. The .csv file is then converted to .arff format using Weka: https://sourceforge.net/projects/weka/. This would be necessary because train_local.py processes the MNIST dataset in .arff format. However, although a .csv file is created, which Weka successfully converts to .arff, I still need to label each image with disease type (1, 2 or 3) and I am running short of time. Therefore, plan B - find another algorithm, capable of processing JPGs directly to a prediction, without the .csv to .arff conversion stages.

train_local_rice_leaf.py is based on Connor Shorten's 'Image Classification Keras Tutorial: Kaggle Dog Breed Challenge' (https://www.kaggle.com/c/dog-breed-identification/overview): https://towardsdatascience.com/image-classification-python-keras-tutorial-kaggle-challenge-45a6332a58b8 This tutorial makes a prediction based on the classification of sample JPGs for two dog breeds (and I have three rice leaf diseases to classify). I omit the coding to rename the JPGs (disease and unique number) as I had already completed this manually in Windows Explorer. train_local_rice_leaf.py imports the leaf image JPG, labels the data by disease, creating train_data[]. The next model and evaluation stage is: simple_cnn_model_evaluation.py. At time of writing, this is a copy of train_local_rice_leaf.py with Connor's 'Simple CNN' script added. However, it requires TensorFlow, which is looking for 'GPU Support', something that is beyond my laptop. Therefore, I need to move to a suitably configured cloud environment before I can finish editing simple_cnn_model_evaluation.py.

The cloud is likely to be Codalab (not done this bit yet): https://codalab.org/ - Update, as there's no obvious way to integrate Codalab with vscode, and this project is meant for people who are not experienced coders nor command line users, then I will look again at using Azure. (The dataset is from UCI and the algorithm from Kaggle, so using Azure will still the criteria of resources from different organisations). 

The JPG images are from the 'Rice Leaf Diseases Data Set', the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Rice+Leaf+Diseases Created by the research of: Jitesh P. Shah, Email: jitesh2k12 '@' gmail.com, Institute: Department of Information Technology, Dharmsinh Desai University,Nadiad-387001, Gujarat, INDIA. Harshadkumar B. Prajapati, Email: prajapatihb.it '@' ddu.ac.in, Institute: Department of Information Technology, Dharmsinh Desai University,Nadiad-387001, Gujarat, INDIA. Vipul K. Dabhi, Email: vipuldabhi.it '@' ddu.ac.in, Institute: Department of Information Technology, Dharmsinh Desai University,Nadiad-387001, Gujarat, INDIA.

Joan Millington, joanmillington13@gmail.com
