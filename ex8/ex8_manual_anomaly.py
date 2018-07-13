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



def estimateMultivariateGaussian(X):
    mu=np.mean(X,axis=0)
    Xmu=X-mu
    sigma2=Xmu.T.dot(Xmu)/X.shape[0]
    return (mu,sigma2)



def multivariateGaussian(X,mu,sigma2):
    n=X.shape[1] # dimensionality of the problem
    Xmu=X-mu
    # check if sigma2 is a 2D array or 1D vector
    if sigma2.ndim==1:
        sigma2=np.diagflat(sigma2)
    # calculate p-values
    XSigX=np.sum(np.dot(Xmu,np.linalg.inv(sigma2))*Xmu,1)
    p=np.exp(-0.5*XSigX) * (2*np.pi)**(-n/2) * np.linalg.det(sigma2)**(-0.5)
    return p



def visualizeFit(X,mu,sigma2):
    # create the mesh grid and calcualte waights for it
    meshvals=np.arange(0,30,.25)
    X1,X2=np.meshgrid(meshvals,meshvals)
    Z=np.hstack((X1.reshape((-1,1)),X2.reshape((-1,1))))
    Z=multivariateGaussian(Z,mu,sigma2).reshape(np.shape(X1))

    # create contour-levels
    mylevels=10.0**np.arange(-20,0,2)	    
    
    # plot
    fig,ax=plt.subplots()
    ax.plot(X[:,0],X[:,1],'bx')
    ax.contour(X1,X2,Z,mylevels)

    return (fig,ax)



def selectThreshold(y_cv,p_cv):
    # initial values
    bestEpsilon=0
    bestF1=-666
    stepsize=(p_cv.max()-p_cv.min())/1000
    evals=np.arange(p_cv.min(),p_cv.max(),stepsize)
    
    for epsilon in evals:
        predictions=(p_cv<epsilon)
        tp=np.sum( predictions &  y_cv)
        fp=np.sum( predictions & ~y_cv)
        fn=np.sum(~predictions &  y_cv)
        if (tp+fp==0) or (tp+fn==0):
            continue
        prec=1.0*tp/(tp+fp)
        rec =1.0*tp/(tp+fn)
        F1=2*prec*rec/(prec+rec)
        if F1>bestF1:
            bestF1=F1
            bestEpsilon=epsilon
    
    return (bestEpsilon,bestF1)

#%% load data

data=scipy.io.loadmat('data/ex8data1.mat')
data.keys()

X=data.get('X')
X_cv=data.get('Xval')
y_cv=data.get('yval').ravel().astype('bool')

# print shapes
print('X:   ',X.shape)
print('X_cv:',X_cv.shape)
print('y_cv:',y_cv.shape)

#%% plot data

plt.plot(X[:,0],X[:,1],'bx')
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)');
plt.show()

#%% apply gaussian fitting

# choose one or the other
mu,sigma2=estimateGaussian(X)
#mu,sigma2=estimateMultivariateGaussian(X)

# evaluate probability of each data point
p=multivariateGaussian(X,mu,sigma2)

#%% plot the Gaussian distribution

fig,ax=visualizeFit(X,mu,sigma2)
fig.show()

#%% find an appropriate threshold via F1-score

p_cv=multivariateGaussian(X_cv,mu,sigma2)
epsilon,F1=selectThreshold(y_cv,p_cv)

#%% visualize the anomalies

fig,ax=visualizeFit(X,mu,sigma2)
indOfAnom   =np.where(p   <epsilon)
indOfAnom_cv=np.where(p_cv<epsilon)
plt.plot(X[indOfAnom[0],0],X[indOfAnom[0],1],'rs',label='anomalies')
ax.legend()
ax.set_title('Main set')
fig.show()

print('Found '+str(indOfAnom[0].size)   +' anomalies in the main set and')
print('      '+str(indOfAnom_cv[0].size)+' anomalies in the CV set')

#%% run the same algorithm for a multi-dimensional case

data=scipy.io.loadmat('data/ex8data2.mat')
data.keys()

X=data.get('X')
X_cv=data.get('Xval')
y_cv=data.get('yval').ravel().astype('bool')

# print shapes
print('X:   ',X.shape)
print('X_cv:',X_cv.shape)
print('y_cv:',y_cv.shape)

# choose one or the other
mu,sigma2=estimateGaussian(X)
#mu,sigma2=estimateMultivariateGaussian(X)
p=multivariateGaussian(X,mu,sigma2)
p_cv=multivariateGaussian(X_cv,mu,sigma2)
epsilon,F1=selectThreshold(y_cv,p_cv)

indOfAnom   =np.where(p   <epsilon)
indOfAnom_cv=np.where(p_cv<epsilon)
print('Found '+str(indOfAnom[0].size)   +' anomalies in the main set and')
print('      '+str(indOfAnom_cv[0].size)+' anomalies in the CV set')










