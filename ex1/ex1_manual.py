#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 11:57:17 2018

@author: Anton Buyskikh

@brief: Linear regression with one and multiple features.
Features normalization.
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



def normalizeFeatures(X,**kwargs):
    if 'mean' in kwargs:
        mean=kwargs['mean']
    else:
        mean=np.mean(X,axis=0)
        
    if 'std' in kwargs:
        std=kwargs['std']
    else:
        std =np.std(X,axis=0)
        
    return np.divide((X-mean),std),mean,std

#%% PART I
# load ex1data1.txt - linear regression with one parameter
    
data1=pd.read_csv("data/ex1data1.txt",names=["X","y"])
x=data1.X.values[:,None]
X=np.hstack((np.ones_like(x),x))
y=data1.y.values

#%% solution via the Gradient Decent and Normal Equation

alpha=0.01
niter=15000
theta_grad,J_hist=getThetaFromGradDecent(X,y,False,alpha,niter)
theta_norm=getThetaFromNormEquation(X,y)

#%% plot the solution via the Gradient Decent

fig, ax = plt.subplots() # create empty figure
plt.plot(x,y,'rx',x,np.dot(X,theta_grad),'b-')
ax.set_xlabel("Population of City in 10,000s")
ax.set_ylabel("Profit in $10,000s")
plt.legend(['Training Data','Linear Regression'])
plt.show()

#%% plot convergence of the Gradient Decent

fig, ax = plt.subplots() # create empty figure
ax.plot(range(niter+1),J_hist,'.-')
ax.set_yscale('log')
ax.set_xlabel("iterations")
ax.set_ylabel("cost function J")
fig.show()

#%% plot the cost function

# calculate values
theta0_vals=np.linspace(-10, 10, 120)
theta1_vals=np.linspace(-1, 4, 100)
J_vals=np.zeros((len(theta1_vals),len(theta0_vals)))
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        J_vals[j][i]=getCost(X,y,[theta0_vals[i],theta1_vals[j]])

# contour plot        
fig = plt.figure()
ax = plt.subplot(111)
plt.contour(theta0_vals,theta1_vals,J_vals,levels=np.linspace(0,600,61))
plt.plot(theta_grad[0],theta_grad[1],'rx')
plt.plot(theta_norm[0],theta_norm[1],'b.')
ax.set_xlabel(r"$\theta_0$")
ax.set_ylabel(r"$\theta_1$")
fig.show()

#%% PART II
# load dataset2
 
data1=pd.read_csv("data/ex1data2.txt",names=["area","nbed","price"])
y=data1.price.values
X=np.hstack((np.ones(len(y))[:,None],data1.area.values[:,None],data1.nbed.values[:,None]))

#%% normalize features

# NB without this section the Gradient Decent does not converge
X[:,1:],X_mean,X_std=normalizeFeatures(X[:,1:])

#%% solution via the Gradient Decent and Normal Equation

alpha=0.1
niter=1500
theta_grad,J_hist=getThetaFromGradDecent(X,y,False,alpha,niter)
theta_norm=getThetaFromNormEquation(X,y)

#%% plot convergence of the Gradient Decent

fig,ax = plt.subplots() # create empty figure
ax.plot(range(50),J_hist[:50],'.-')
#ax.set_yscale('log')
ax.set_xlabel("iterations")
ax.set_ylabel("cost function J")
fig.show()

#%% prediction

predict=np.array([1,1650,3],dtype='float')[None,:]
predict[:,1:],_,_=normalizeFeatures(predict[0,1:],mean=X_mean,std=X_std)
price=hypothesisFunc(predict,theta_norm)

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): \n',
       price)
