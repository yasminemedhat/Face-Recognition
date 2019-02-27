# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:32:46 2019

@author: lenovo
"""
import os
import numpy as np
from PIL import Image




# function to prepare the data set 
def prepare_data_set(path):
 
    #list to hold all subject faces
    faces = []
    #list to hold labels for each subject
    labels = []
    l = 1
    #loop through each dir and extract images and labels
    for root , dirnames , filenames in os.walk(path):
        #loop through each subject directory
        for subdirname in dirnames:
            subject_path = os.path.join(root , subdirname)
 
            #go through each image convert it into a numpy array and store it
            for filename in os.listdir(subject_path):
                img = Image.open(os.path.join(subject_path , filename)).convert('L') 
                faces.append(np.asarray(img).ravel())
                labels.append(l)
               
            l = l + 1
    return faces, labels

#function to split data into training set and tests set
def split_data(faces , labels):
    #two lists to hold training data and their labels
    train_faces_set = []
    train_label_set = []
    #two lists to hold test data and their labels
    test_faces_set = []
    test_label_set = []
    i = 0
    #loop over data, if odd append to training else append to test
    for x, y in zip(faces, labels):
        if i % 2 == 0:
            train_faces_set.append(x)
            train_label_set.append(y)
        else:
            test_faces_set.append(x)
            test_label_set.append(y)
        i = i + 1
    return np.asarray(train_faces_set), np.asarray(train_label_set), np.asarray(test_faces_set), np.asarray(test_label_set)



        

def main():
    faces, labels = prepare_data_set(folder_name)
    trainf, trainl, testf, testl = split_data(faces, labels)
    