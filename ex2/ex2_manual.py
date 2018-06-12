#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 09:34:37 2018

@author: Anton Buyskikh

@brief: Logistic regression with multiple features.
...
"""

#%% libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit

#from scipy.optimize import minimize
import scipy

#%% functions

def plotData(X,y):
    pos=X[np.where(y==1)]
    neg=X[np.where(y==0)]
    fig,ax=plt.subplots()
    ax.plot(pos[:,0],pos[:,1],"k+",neg[:,0],neg[:,1],"yo")
    return (fig,ax)



def h(X,theta):
    # hypothesis
    return 1.0/(1.0+np.exp(-X@theta))



def getCostAndGrad(X,y,theta):
    m,n=X.shape
    J=np.mean(-y*np.log(h(X,theta))-(1-y)*(np.log(1-h(X,theta))))
    grad=(h(X,theta)-y)@X/m
    return (J,grad)



def predict(X,theta):
    print("Doublecheck this one")
    return np.where(h(X,theta)>0.5,1,0)

#%% load ex2data1.txt - logical regression with two parameters
    
data1=pd.read_csv('data/ex2data1.txt',names=['exam1','exam2','score'])
y=np.asarray(data1.score)
X=np.hstack((np.ones_like(y)[:,None],np.asarray(data1[["exam1","exam2"]])))

#%% visualize data

fig,ax=plotData(X[:,1:],y)
ax.legend(['Admitted','Not admitted'])
fig.show()

#%% get const functions

theta=np.zeros(3)
cost,grad=getCostAndGrad(X,y,theta)
print('Cost at initial theta (zeros): \n', cost)
print('Gradient at initial theta (zeros): \n',grad)

#%% solution via the Gradient Decent and Normal Equation



#%% plot the solution via the Gradient Decent



#%% plot convergence of the Gradient Decent



#%% plot the cost function



#%% load dataset2
 

#%% normalize features


#%% solution via the Gradient Decent and Normal Equation


#%% plot convergence of the Gradient Decent



#%% prediction


