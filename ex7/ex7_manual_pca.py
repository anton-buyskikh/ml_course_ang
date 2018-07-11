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
import random

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
    X_rec=Z[:,:K].dot(U[:,:K].T)
    return X_rec



def displayData(X):
    m,n=X.shape
    # define the grid with images
    rows=int(m**0.5)
    cols=int(m**0.5)
    # define the size of images
    sz1=int(n**0.5)
    sz2=int(n**0.5)
    
    # plot
    fig,ax=plt.subplots(rows,cols,sharex=True,sharey=True)
    img_num=0
    for i in range(rows):
        for j in range(cols):
            img=X[img_num,:].reshape(sz1,sz2).T
            ax[i,j].imshow(img,cmap='gray')
            img_num+=1

    return (fig,ax)

#%% get data1 --- simple dimension reduction

data1=scipy.io.loadmat("data/ex7data1.mat")
X=data1.get("X")
plt.cla()
plt.plot(X[:,0],X[:,1],'bo')
plt.show()

#%% normalize and factorize data with PCA

X_norm,mu,sigma=normalizeFeatures(X)
U,S,V=pca(X_norm)

#%% recover data

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
ax.set_title('Origianl axes')
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
ax.set_title('Normalized axes')
ax.axis([-3,3,-3,3])
ax.set_aspect('equal')
fig.show()

#%% get data with faces and plot a random sample

data2=scipy.io.loadmat("data/ex7faces.mat")
X=data2.get("X")

sample=random.sample(range(X.shape[0]),25)
face_grid,ax=displayData(X[sample,:])
face_grid.show()

#%% normalize and factorize data with PCA

X_norm,mu,sigma=normalizeFeatures(X)
U,S,V=pca(X_norm)

#%% display the largest principal components --- eigenfaces? :)

face_grid,ax=displayData(U[:,:36].T)
face_grid.show()

#%% recover data

K=100 # number of principal axes
Z=projectData(X_norm,U,K)
X_norm_rec=recoverData(Z,U,K)
X_rec=unnormalizeFeatures(X_norm_rec,mu,sigma)

#%% plot original and recovered images together

sample=random.sample(range(X.shape[0]),25)
face_grid,ax=displayData(X[sample,:])
face_grid.show()

face_grid,ax=displayData(X_rec[sample,:])
face_grid.show()











