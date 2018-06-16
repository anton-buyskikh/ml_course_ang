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
    return 1.0/(1.0+np.exp(-X@theta))



def getCostAndGrad(theta,X,y):
    # Returns the cost function and its gradient
    # see Sec. 1.2.2
    m,n=X.shape
    J=(-y*np.log(h(X,theta))-(1-y)*(np.log(1-h(X,theta)))).mean()
    grad=(h(X,theta)-y)@X/m
    return (J,grad)



def getCostAndGradReg(theta,X,y,lam_par):
    # Returns the regularized cost function and its gradient
    # see Sec. 2.3
    J,grad=getCostAndGrad(theta,X,y)
    m,n=X.shape
    J+=lam_par*(theta[1:]**2).sum()/2/m
    grad[1:]+=lam_par/m*theta[1:]
    return (J,grad)



def predict(X,theta,threshold=0.5):
    return np.where(h(X,theta)>threshold,1,0)



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
        ytarget=np.array(y==k_list[ik]).astype(int)        
        theta_init=theta[:,ik]

        res=scipy.optimize.minimize(getCostAndGradReg,\
                                    theta_init,\
                                    args=(X,ytarget,lam_par),\
                                    method='Newton-CG',\
                                    tol=1e-3,\
                                    jac=True,\
                                    options={'maxiter':10,'disp':True})
        
        theta[:,ik]=res.x
        
    return theta



def predictOneVsAllAccuracy(theta,X):
    probs=X@theta
    predict=np.argmax(probs,axis=1)    
    return predict

#%% PART I
# load data and weights from Matlab files
data=scipy.io.loadmat('data/ex3data1.mat')
weights=scipy.io.loadmat('data/ex3weights.mat')

print('data    keys: ',data.keys())
print('weights keys: ',weights.keys())

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

sample=np.random.randint(0,X.shape[0],100)
fig,ax=displayData(X[sample,1:],y[sample])
fig.show()

#%% vectorized logistic regression

lam_par=1.0
theta_opt=oneVsAll(X,y,range(10),lam_par)

#%% accuracy

predictions=predictOneVsAllAccuracy(theta_opt,X)
accuracy=np.mean(y==predictions)*100
print("Training Accuracy with logit: ",accuracy,"%")

#%% PART II


