# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:32:46 2019

@author: lenovo
"""
import os
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from PCA import PCA

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

#function to split data into training set and tests set 50-50
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
    return train_faces_set, test_faces_set, train_label_set, test_label_set

#split data 70-30 from each subject
def split_data_7_3(faces, labels):
     #two lists to hold training data and their labels
     train_faces_set = []
     train_label_set = []
     #two lists to hold test data and their labels
     test_faces_set = []
     test_label_set = []
     i = 0
     for x, y in zip(faces, labels):
         if i < 7:
             train_faces_set.append(x)
             train_label_set.append(y)
         elif i >= 7:
             test_faces_set.append(x)
             test_label_set.append(y)
         i = i + 1
         if(i >= 10):
             i = 0

     return train_faces_set, test_faces_set, train_label_set, test_label_set

#perform LDA on training set D and calculating the new direction with k components number
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
    eigenVals, eigenVecs = np.linalg.eigh(np.dot(np.linalg.inv(Sw), Sb))
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
    knn = KNeighborsClassifier(n_neighbors=k, weights = 'unifrom')
    knn.fit(train_faces, train_labels)
    #get the prediction for each image in the test set
    predictions = knn.predict(test_faces)
    #get the accuracy of the prediction
    accuracy = metrics.accuracy_score(test_labels, predictions)
    
    return accuracy * 100
 
#plot accuracy against different number of neighbors in KNN classification
def accuracy_against_Kneighbors(x_train, x_test, y_train, y_test):
    #change number of neighors(1,3,5,7) and compute accuracy
    accuracy = []
    K = []
    for k in range(1, 8, 2):
        accuracy.append(KNN(x_test, y_test, x_train, y_train,k))
        K.append(k)
    #plot the accuracy against number of neighbours in KNN
    plt.plot(K, accuracy, 'ro')
    plt.axis([1, 10, 0, 100])
    plt.legend()
    plt.ylabel('accuracy (%)')
    plt.xlabel('numbers of neighbors')
    plt.show()
    

def main():
    folder_name = r"E:\Documents\Term 8\Pattern Recognition\assignments\assignment 1\Data"
    faces, labels = prepare_data_set(folder_name)
    faces = np.asarray(faces)
    labels = np.asarray(labels)
    #split data to training and test sets
    train_faces, test_faces, train_labels, test_labels = split_data(faces, labels)
    train_faces = np.asarray(train_faces)
    train_labels = np.asarray(train_labels)
    test_faces = np.asarray(test_faces)
    test_labels = np.asarray(test_labels)
    #perform LDA on training set and get the best direction
    W = LDA(train_faces, train_labels, 39)
    #project Training set
    newtrain = project(train_faces, W)
    #project test set on the same direction
    newtest =  project(test_faces, W)
    #compute accuracy
    accuracyLDA = KNN(newtest, test_labels, newtrain, train_labels,1)
    print("accuracy of LDA: ", accuracyLDA,"%")
   
    #plot accuracy with different number of neighbors
    accuracy_against_Kneighbors(train_faces, test_faces, train_labels, test_labels)
    
    #split data matrix differently
    #split as 0.7 training and 0.3 testing
    X_train, X_test, y_train, y_test = split_data_7_3(faces, labels)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    #perform LDA on training set and get the best direction
    z = LDA(X_train, y_train, 39)
    #project Training set
    new_X_train = project(X_train, z)
    #project test set on the same direction
    new_X_test =  project(X_test, z)
    #compute accuracy
    accuracyLDA2 = KNN(new_X_test, y_test , new_X_train, y_train,1)
    print("accuracy of LDA (7-3 train-test): ", accuracyLDA2,"%")
    
    #plot accuracy with different number of neighbors
    accuracy_against_Kneighbors(new_X_train, new_X_test, y_train, y_test)
    
     #PCA
    print("\n\nPCA:\n")
    alpha=np.array([0.8,0.85,0.9,0.95])
    reducedD,vecs=PCA(train_faces,alpha)  #number of dimensions for each alpha and eigenvector to use
    for alph,dim in zip(alpha,reducedD):
        print("For alpha= ",alph,"\nDimension= ",int(dim)+1)
        reduced=vecs[:,:int(dim)+1]
        Atrain=reduced.T.dot(train_faces.T)
        Atrain=Atrain.T
        Atest=reduced.T.dot(test_faces.T)
        Atest=Atest.T
        print("accuracy= ", KNN(Atest, test_labels, Atrain, train_labels,1),"%\n\n") #nearest neighbour and accuracy per alpha

    #plot accuracy for PCA with different number of neighbors
    accuracy_against_Kneighbors(Atrain, Atest, train_labels, test_labels)
    
    #PCA on (7-3) train-test data

