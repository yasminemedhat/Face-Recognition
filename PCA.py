# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 20:05:37 2019

@author: Loujaina
"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def PCA(D , alpha):
    #compute mean, axis=1-->
    global mean
    mean=np.mean(D,axis=0)
    print("Mean=\n",mean)
    #Centre the data
    #no need to transpose the mean because
    #default shape is a vector
    Z=D-mean
    
    #Compute covariance matrix
    #rowvar=False --> each column reps a variable
    #bias=Ture --> normalization is by N
    global cov
    cov=np.cov(Z,rowvar=False,bias=True)
    print("cov=\n",cov)
    #EigenValues and EigenVecs:
    global vals,vecs,reduced
    vals, vecs = np.linalg.eigh(cov)
   
    #Sorting of EigenVectors based on eigen values (descending order):
    index=vals.argsort()[::-1] #return indices used for sorting
    vals=vals[index]
    vecs=vecs[:,index]
    print("vals=\n",vals)
    print("vecs=\n",vecs)

    
    #Explained Variance -> choose dimensionality
    sumVals=np.sum(vals)
    fractionVals=0
    i=-1; #will use first i vecs
    print(vals.size)
    while (round(fractionVals/float(sumVals),2)<alpha and i<vals.size-1):
        i+=1
        fractionVals+=vals[i]
        print(i,fractionVals/float(sumVals))
    else:
        print("Needed r (starting from zero)= ",i) 
        reduced=vecs[:,:i+1]
        
    return reduced
        
    
#Testing using iris dataset 
def main():
    
    iris=datasets.load_iris()
    x=iris.data[:150,:3]
    reduced=PCA(x,1)
    A=reduced.T.dot(x.T)
    A=A.T
    plt.scatter(A[:,0],A[:,1])
    plt.show()
    