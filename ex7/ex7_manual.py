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
import random

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



def runKMeans(X,centroids,max_iters,isHistory=False):
    m,n=X.shape
    K=centroids.shape[0]

    # save the history of iterations
    if isHistory:
        history=np.full((K,n,max_iters),np.nan)
        history[:,:,0]=centroids

    for iter in range(max_iters):
        indc=findClosestCentroids(X,centroids)
        centroids=computeCentroids(X,indc,K)
        if isHistory:
            history[:,:,iter]=centroids

    if isHistory:
        return (centroids,indc,history)
    else:
        return (centroids,indc)

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

#%% K-means Clustering

# pick random data points as K centroids
K=3
sample=random.sample(range(m),K)
centroids=X[sample,:]

# run the search
max_iters=10
centroids,indc,history=runKMeans(X,centroids,max_iters,isHistory=True)

#%% display the result

fig,ax=plt.subplots()
for k in range(K):
    ax.plot(X[indc==k,0],X[indc==k,1],'.',label='final cluster '+str(k))
    ax.plot(history[k,0,:],history[k,1,:],':xk')    
ax.plot(centroids[:,0],centroids[:,1],'ok',label='final centroids')
ax.legend(loc='best')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
fig.show()



















