#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 11:57:17 2018

@author: anton
"""

#%% libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import timeit

#%% functions

def warmUpExercise():
    return np.eye(5,dtype='float')



# linear regression hypothesis
def hypothesisFunc(X,theta):
    return X@theta



def getThetaFromNormEquation(X,y):
    return np.linalg.inv(X.T@X)@X.T@y



def getThetaFromGradDecent(X,y,isRandTheta,alpha,niter):    
    m,n=X.shape
    if isRandTheta:
        theta=np.random.rand(n)
    else:
        theta=np.zeros(n,dtype='float')
    
    J_hist=np.full(niter+1,np.nan)
    J_hist[0]=getCost(X,y,theta)
    
    for it in range(niter):      
        theta-=alpha*(hypothesisFunc(X,theta)-y)@X/m
        J_hist[it+1]=getCost(X,y,theta)
    
    return theta,J_hist


    
def getCost(X,y,theta):
    return np.mean((hypothesisFunc(X,theta)-y)**2)/2

#%% load data
    
data=pd.read_csv("data/ex1data1.txt",names=["X","y"])
x=data.X.as_matrix()[:,None]
X=np.hstack((np.ones_like(x),x))
y=data.y.as_matrix()

#%% solution via the Gradient Decent

niter=15000
alpha=0.01
theta_grad,J_hist=getThetaFromGradDecent(X,y,False,alpha,niter)

#%% plot the solution via the Gradient Decent

fig, ax = plt.subplots() # create empty figure
plt.plot(x,y,'rx',x,np.dot(X,theta_grad),'b-')
ax.set_xlabel("Population of City in 10,000s")
ax.set_ylabel("Profit in $10,000s")
plt.legend(['Training Data','Linear Regression'])
plt.show()



#%% solution via the Normal Equation


#%% plot the solution via the Normal Equation


#%% convergence of the Gradient Decent method

fig, ax = plt.subplots() # create empty figure
ax.plot(range(niter+1),J_hist,'.-')
ax.set_xlabel("iter")
ax.set_ylabel("J")
fig.show()






