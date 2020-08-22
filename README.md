Project Dissertation

This code is part of my project dissertation on 'Artificial Intelligence and Data Commons'.

The aim is to test whether it is possible for someone with little or no python coding experience to combine:

    a publically available algorithm from one source,
    with a publically available dataset from another source,
    and run it in a cloud environment (without prior link to the algorithm or dataset)
    and deliver a prediction (it doesn't have to be a good prediction, as machine learning skills are not in scope).

The dataset chosen is the 'Rice Leaf Diseases Data Set', from the UCI Machine Learning Repository: 
https://archive.ics.uci.edu/ml/datasets/Rice+Leaf+Diseases 
Created by the research of: Jitesh P. Shah, Email: jitesh2k12 '@' gmail.com, Institute: Department of Information Technology, Dharmsinh Desai University,Nadiad-387001, Gujarat, INDIA. Harshadkumar B. Prajapati, Email: prajapatihb.it '@' ddu.ac.in, Institute: Department of Information Technology, Dharmsinh Desai University,Nadiad-387001, Gujarat, INDIA. Vipul K. Dabhi, Email: vipuldabhi.it '@' ddu.ac.in, Institute: Department of Information Technology, Dharmsinh Desai University,Nadiad-387001, Gujarat, INDIA.

The 'simple_cnn_model_evaluation.py' algorithm is based on Connor Shorten's 'Image Classification Keras Tutorial Kaggle Dog Breed Challenge': 
https://towardsdatascience.com/image-classification-python-keras-tutorial-kaggle-challenge-45a6332a58b8 
https://www.kaggle.com/c/dog-breed-identification/overview
Connor's tutorial makes a prediction based upon the classification of sample JPGs for two dog breeds (and I have three rice leaf diseases to classify). I omit the coding to rename the JPGs (disease and unique number) as I had already completed this manually in Windows Explorer. My adapted script imports the rice leaf image JPGs, creates training data, against which the model is run and evaluated. Processing the JPGs requires a graphics processing unit and therefore this script is run on an Azure virtual machine, within an Azure ML environment. 

The purpose of 'image_display_grid.py' is to provide an image of the rice leaves for inclusion within my dissertation text. It is based upon the image plot code in 'train_local.py' from Dmitry Soshnikov's 'AzureMlStarter', see blog: 
https://soshnikov.com/azure/best-way-to-start-with-azureml/. 
I have edited the script to display a 10 x 12 block of rice leaf images. The script runs on my laptop with a local environment.

I also used Dmitry's blog for guidance on how to link Visual Studio Code (my IDE) with the Microsoft Azure Machine Learning Portal (for cloud provision). The virtual machine created provides the GPU capability necessary for 'simple_cnn_model_evaluation.py' JPG processing.

Joan Millington, joanmillington13@gmail.com
August 2020
