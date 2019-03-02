# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:32:46 2019
@author: lenovo
"""
import os
import numpy as np
from PIL import Image
from PCA import PCA  
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics




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

def KNN(test_faces, test_labels, train_faces, train_labels, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_faces, train_labels)
    #get the prediction for each image in the test set
    predictions = knn.predict(test_faces)
    #get the accuracy of the prediction
    accuracy = metrics.accuracy_score(test_labels, predictions)
    
    return accuracy * 100

        

def main():
    faces, labels = prepare_data_set("Data")
    trainf, trainl, testf, testl = split_data(faces, labels)
    
    #PCA
    print("\n\nPCA:\n")
    alpha=np.array([0.8,0.85,0.9,0.95])
    reducedD,vecs=PCA(trainf,alpha)  #number of dimensions for each alpha and eigenvector to use
    for alph,dim in zip(alpha,reducedD):
        print("For alpha= ",alph,"\nDimensions= ",int(dim)+1)
        reduced=vecs[:,:int(dim)+1]
        Atrain=reduced.T.dot(trainf.T)
        Atrain=Atrain.T
        Atest=reduced.T.dot(testf.T)
        Atest=Atest.T
        print("accuracy= ", KNN(Atest, testl, Atrain, trainl,1),"%\n\n") #nearest neighbour and accuracy per alpha
        
        
        
    
    
