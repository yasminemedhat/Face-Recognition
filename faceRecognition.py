# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:32:46 2019

@author: lenovo
"""
import os, sys
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

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

#perform LDA on training set D and calculatin the new direction with k components number
def LDA(D, labels, k):
    D = np.asarray(D)
    labels = np.asarray(labels)
    #get shape of the data matrix
    n, d = D.shape
    #get number of classes
    classes = np.unique(labels)
    #total mean
    meanTotal = D.mean(axis=0)
    #between class scatter matrix
    Sb = np.zeros((d,d),dtype=np.float32)
    #within class scatter matrix
    Sw = np.zeros((d,d),dtype=np.float32)
    print("calculating means, between class scatter and within class scatter matrices...")
    #calc mean and within class scatter for each class
    for i in classes:
        #get each subject class
        Di = D[np.where(labels==i)[0],:]
        #calc its mean
        meani = Di.mean(axis=0)
        Sw = Sw + np.dot((Di - meani).T,(Di - meani))
        Sb = Sb + n * np.dot((meani - meanTotal).T,(meani - meanTotal))
    print("calculating the eigen values and eigen vectors..")
    eigenVals, eigenVecs = np.linalg.eig(np.dot(np.linalg.inv(Sw), Sb))
    #get indexes of descendigly sorted eigen values array
    idx = eigenVals.argsort()[::-1][:k]
    #get the k eigenvectors accordingly
    eigenVecs = np.array(eigenVecs[:,idx].real,dtype= np.float32)
    
    return eigenVecs.T

#return the projection of x on w    
def project(x, w):
    x = np.asarray(x)
    w = np.asarray(w)
    
    return np.dot(w,x.T).T

#perform KNN on the test set and return the accuracy
def KNN(test_faces, test_labels, train_faces, train_labels, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_faces, train_labels)
    #get the prediction for each image in the test set
    predictions = knn.predict(test_faces)
    #get the accuracy of the prediction
    accuracy = metrics.accuracy_score(test_labels, predictions)
    
    return accuracy * 100
    

def main():
    folder_name = r"E:\Documents\Term 8\Pattern Recognition\assignments\assignment 1\Data"
    faces, labels = prepare_data_set(folder_name)
    #split data to training and test sets
    trainf, trainl, testf, testl = split_data(faces, labels)
    #perform LDA on training set and get the best direction
    W = LDA(trainf, trainl, 39)
    #project Training set
    newtrain = project(trainf, W)
    #project test set on the same direction
    newtest =  project(testf, W)
    #compute accuracy
    accuracyLDA = KNN(newtest, testl, newtrain, trainl,1)
    print("accuracy of LDA: ", accuracyLDA,"%")
    
    
#if __name__ == "__main__":
   # main()