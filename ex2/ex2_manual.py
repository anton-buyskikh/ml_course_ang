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



def getCostAndGrad(theta,X,y):
    # Returns the cost function and its gradient Sec. 1.2.2
    m,n=X.shape
    J=np.mean(-y*np.log(h(X,theta))-(1-y)*(np.log(1-h(X,theta))))
    grad=(h(X,theta)-y)@X/m
    return (J,grad)



def predict(X,theta):
    return np.where(h(X,theta)>0.5,1,0)

#%% PART I
# load ex2data1.txt - logical regression with two parameters
    
data1=pd.read_csv('data/ex2data1.txt',names=['exam1','exam2','score'])
y=np.asarray(data1.score)
X=np.hstack((np.ones_like(y)[:,None],np.asarray(data1[["exam1","exam2"]])))

#%% visualize data

fig,ax=plotData(X[:,1:],y)
ax.legend(['Admitted','Not admitted'])
fig.show()

#%% get const functions

theta_init=np.zeros(3)
cost,grad=getCostAndGrad(theta_init,X,y)
print('Cost at theta_init=[0 0 0]: \n', cost)
print('Gradient at theta_init=[0 0 0]: \n',grad, '\n')

#%% solution via the optimization algorithm

res=scipy.optimize.minimize(getCostAndGrad,
                            theta_init,
                            args=(X,y),
                            method='Newton-CG',
                            jac=True,
                            options={'maxiter':400,'disp':True})

theta_opt=res.x
print('theta_opt: ',theta_opt)
print('Cost at theta_opt: ',res.fun, '\n')

#%% plot the solution from the previous block

#NB: there should be a nicer way!!!
x1=np.linspace(30,100,61)
x2=(-theta_opt[0]-theta_opt[1]*x1)/theta_opt[2]
fig,ax=plotData(X[:,1:],y)
ax.plot(x1,x2,'-r')
ax.legend(['Admitted','Not admitted','Decision boundary'])
fig.show()

#%% prediction and accuracy

print('For a student with scores (45,85), admission probability is',
      h(np.array([1,45,85]),theta_opt))

# Accuracy on the training set with the optimal parameters
print('Train Accuracy: ',np.mean(predict(X,theta_opt)==y)*100, '\n')

#%% PART II





