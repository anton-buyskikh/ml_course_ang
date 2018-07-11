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



def unnormalizeFeatures(X,mean,std):
    return np.multiply(X,std)+mean



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

#%% get data1 --- simple dimension reduction

data1=scipy.io.loadmat("data/ex7data1.mat")
X=data1.get("X")
plt.cla()
plt.plot(X[:,0],X[:,1],'bo')
plt.show()

#%% normalize and compress data with PCA

X_norm,mu,sigma=normalizeFeatures(X)
U,S,V=pca(X_norm)

#%% recover the data

K=1 # number of principal axes
Z=projectData(X_norm,U,K)
X_norm_rec=recoverData(Z,U,K)
X_rec=unnormalizeFeatures(X_norm_rec,mu,sigma)

#%% plot on original scales

fig,ax=plt.subplots()
for i in range(X.shape[0]):
    ax.plot([X[i,0],X_rec[i,0]],[X[i,1],X_rec[i,1]],':k')
ax.plot(X[:,0],     X[:,1]    ,'b.',label='original')
ax.plot(X_rec[:,0], X_rec[:,1],'r.',label='reduced')
ax.legend()
ax.axis([1,7,2,8])
ax.set_aspect('equal')
fig.show()

# NOTE: Since sigmas are slightly different, it's not exacly a 90degree
#       projection. However in the normalized units it is (see below)

#%% plot on normalized scales

fig,ax=plt.subplots()
for i in range(X.shape[0]):
    ax.plot([X_norm[i,0],X_norm_rec[i,0]],[X_norm[i,1],X_norm_rec[i,1]],':k')
ax.plot(X_norm[:,0],     X_norm[:,1]    ,'b.',label='original')
ax.plot(X_norm_rec[:,0], X_norm_rec[:,1],'r.',label='reduced')
ax.legend()
ax.axis([-3,3,-3,3])
ax.set_aspect('equal')
fig.show()





















