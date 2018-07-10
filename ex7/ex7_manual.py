#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 17:56:27 2018

@author: Anton Buyskikh
@brief: Unsupervised machine learning. 
K-means clustering.
"""


#%% libraries

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

#%% functions

def findClosestCentroids(X,centroids):
    m,n=X.shape
    indc=np.full(m,np.nan)
    for im in range(m):
        dist2clust=np.sum(((X[im,:]-centroids)**2),axis=1)
        indc[im]=dist2clust.argmin()
    return indc



def computeCentroids(X,indc,K):
    m,n=X.shape
    centroids=np.full((K,n),np.nan)
    for k in range(K):
        centroids[k]=np.mean(X[indc==k,:],axis=0)
    return centroids

#%% PART I: 
# get data

data1=scipy.io.loadmat("data/ex7data2.mat")
X=data1.get("X")
m,n=X.shape

#%% test how code components work

# initial guess
centroids=np.array([[3, 3], [6, 2], [8, 5]],dtype='float')
K=centroids.shape[0]  # number of centroids
indc=findClosestCentroids(X,centroids)
centroids=computeCentroids(X,indc,K)






















