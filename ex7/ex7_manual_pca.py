#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 23:55:06 2018

@author: Anton Buyskikh
@brief: Unsupervised machine learning. 
Principal Component Analysis. Dimensionality reduction.
"""
#%% libraries

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

#%% functions

def normalizeFeatures(X,**kwargs):
    if 'mean' in kwargs:
        mean=kwargs['mean']
    else:
        mean=np.mean(X,axis=0)
        
    if 'std' in kwargs:
        std=kwargs['std']
    else:
        std =np.std(X,axis=0)
        
    return (np.divide((X-mean),std),mean,std)



def pca(X):
	covar=np.dot(X.T,X)/X.shape[0]
	U,S,V=np.linalg.svd(covar)
	return (U,S,V)



def projectData(X,U,K):
	return X.dot(U[:,:K])    



def recoverData(Z,U,K):
	X_rec=np.zeros((Z.shape[0],U.shape[1]))
	for i in range(Z.shape[0]):
		for j in range(U.shape[1]):
			recovered_j=np.dot(Z[i,:].T,U[j,:K])
			X_rec[i,j]=recovered_j
	return X_rec

#%% get data

data1=scipy.io.loadmat("data/ex7data1.mat")
X=data1.get("X")
plt.cla()
plt.plot(X[:,0],X[:,1],'bo')
plt.show()

#%% normalize and compress data with PCA

X_norm,mu,sigma=normalizeFeatures(X)
U,S,V=pca(X_norm)

#%% recover the data

K=1
Z=projectData(X_norm,U,K)
X_rec=recoverData(Z,U,K)

#%% plot 

plt.cla()
plt.plot(X_norm[:,0], X_norm[:,1],'bo',label='original')
plt.plot(X_rec[:,0], X_rec[:,1], 'rx',label='reduced')
plt.legend()
plt.show()





















