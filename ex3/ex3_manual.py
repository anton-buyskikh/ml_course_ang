#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 10:35:26 2018

@author: Anton Buyskikh

@brief: Logistic regression with multi-class classification.
Neural network method.
"""

#%% libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import scipy

#%% functions

def h(X,theta):
    # hypothesis function
    # by default we use sigmoid
    return 1.0/(1.0+np.exp(-X.dot(theta)))



def getCostReg(theta,X,y,lam_par):
    m,n=X.shape
    J=(-y*np.log(h(X,theta))-(1-y)*(np.log(1-h(X,theta)))).mean()
    J+=lam_par*(theta[1:]**2).sum()/2/m
    return J



def getGradReg(theta,X,y,lam_par):
    m,n=X.shape
    grad=(h(X,theta)-y).dot(X)/m
    grad[1:]+=lam_par/m*theta[1:]
    return grad



def displayData(X,y):
    n1,n2=5,5
    fig,ax=plt.subplots(n1,n2,sharex=True,sharey=True)
    img_num=0
    for i in range(n1):
        for j in range(n2):
            # Convert column vector into 20x20 pixel matrix
            # You have to transpose to display correctly
            img=X[img_num,:].reshape(20,20).T
            ax[i][j].imshow(img,cmap='gray')
            img_num+=1
    return (fig,ax)



def oneVsAll(X,y,k_list,lam_par):
    m,n=X.shape
    K=len(k_list) # number of classes
    theta=np.zeros((n,K))
    
    for ik in k_list:
        print('target k='+str(k_list[ik]))
        res=scipy.optimize.minimize(getCostReg,\
                                    theta[:,ik],\
                                    args=(X,(y==k_list[ik])*1,lam_par),\
                                    method='Newton-CG',\
                                    tol=1e-3,\
                                    jac=getGradReg,\
                                    options={'maxiter':10,'disp':True})        
        theta[:,ik]=res.x
        
    return theta



def predictOneVsAllAccuracy(theta,X):
    probs=h(X,theta)
    predict=np.argmax(probs,axis=1)
    return predict

#%% PART I
# load data from Matlab files
data=scipy.io.loadmat('data/ex3data1.mat')
print('data    keys: ',data.keys())

#%% extract data

y=np.asarray(data['y'],dtype='int').ravel()
y[y==10]=0
#y=tuple(y)
# Add constant for intercept
X=np.c_[np.ones((data['X'].shape[0],1)),np.asarray(data['X'])]

theta1,theta2=weights['Theta1'],weights['Theta2']

print('Shapes of variables:')
print('X: {} (with identity)'.format(X.shape))
print('y: {}'.format(y.shape))
print('theta1: {}'.format(theta1.shape))
print('theta2: {}'.format(theta2.shape))
print('\n')

#%% visualize data

sample=np.random.randint(0,X.shape[0],25)
fig,ax=displayData(X[sample,1:],y[sample])
fig.show()

#%% vectorized logistic regression

lam_par=1.0
theta_opt=oneVsAll(X,y,range(10),lam_par)

#%% prediction

predictions_1vsAll=predictOneVsAllAccuracy(theta_opt,X)
accuracy_1vsAll=np.mean(y==predictions_1vsAll)*100
print("Training Accuracy with scheme: ",accuracy_1vsAll,"%\n")

#%% PART II
# load weights from Matlab files

weights=scipy.io.loadmat('data/ex3weights.mat')
print('weights keys: ',weights.keys())
theta1=np.asarray(weights['Theta1'])
theta2=np.asarray(weights['Theta2'])
print('Theta1: ',Theta1.shape)
print('Theta2: ',Theta2.shape)

#%% prediction

a1=X
a2=h(a1,theta1.T)
a2=np.hstack((np.ones(a2.shape[0])[:,None],a2))
a3=h(a2,theta2.T)

predictions_NN=np.mod(np.argmax(a3,axis=1)+1,10)
accuracy_NN=np.mean(y==predictions_NN)*100
print("Training Accuracy with Nequral Network: ",accuracy_NN,"%\n")

#%% timeit
# test of different contraction schemes

timeit.timeit(stmt='h(X,theta_opt)',setup='from __main__ import h,X,theta_opt',\
                    number=100)


