#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 09:48:35 2018

@author: Anton Buyskikh
@brief: 
...
"""
#%% libraries

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random

#%% functions

def estimateGaussian(X):
	return (np.mean(X,axis=0),np.var(X,axis=0))




def multivariateGaussian(X,mu,sigma2):
    n=X.shape[1] # dimensionality of the problem
    X=X-mu
    # check if sigma2 is a 2D array or 1D vector
    if sigma2.ndim==1:
        sigma2=np.diagflat(sigma2)
    # calculate p-values
    XSigX=np.sum(np.dot(X,np.linalg.inv(sigma2))*X,1)
    p=np.exp(-0.5*XSigX)*((2*np.pi)**n * np.linalg.det(sigma2))**(-0.5)
    return p




def visualizeFit(X,mu,sigma2):
    # create the mesh grid and calcualte waights for it
    meshvals=np.arange(0,30,.25)
    X1,X2=np.meshgrid(meshvals,meshvals)
    Z=np.hstack((X1.reshape((-1,1)),X2.reshape((-1,1))))
    Z=multivariateGaussian(Z,mu,sigma2).reshape(np.shape(X1))

    # create contour-levels
    mylevels=np.array([10.0**i for i in np.arange(-20,0,3)])
	
    # plot
    fig,ax=plt.subplots()
    ax.plot(X[:,0],X[:,1],'bx')
    ax.contour(X1,X2,Z,mylevels)

    return (fig,ax)

#%% load data

data=scipy.io.loadmat('data/ex8data1.mat')
data.keys()

X=data.get('X')
X_cv=data.get('Xval')
y_cv   =data.get('yval').ravel()

# print shapes
print('X:   ',X.shape)
print('X_cv:',X_cv.shape)
print('y_cv:',y_cv.shape)

#%% plot X

plt.plot(X[:,0], X[:,1],'bx')
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)');
plt.show()

#%% apply gaussian fitting

mu,sigma2=estimateGaussian(X)

# test run
p=multivariateGaussian(X, mu, sigma2)

#%% plot

fig,ax=visualizeFit(X,mu,sigma2)
fig.show()









