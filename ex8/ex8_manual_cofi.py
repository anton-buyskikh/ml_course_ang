#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:32:59 2018

@author: Anton Buyskikh
@brief: Recommender Systems.
...
"""
#%% libraries

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random

#%% functions

def flattenParams(X,Theta):
    return np.concatenate((X.flatten(),Theta.flatten()))



def reshapeParams(XTheta,nm,nu,nf):
    # X:     (nm x nf)
    # Theta: (nu x nf)
    assert XTheta.size==int(nm*nf+nu*nf)
    X    =XTheta[:int(nm*nf) ].reshape((nm,nf))
    Theta=XTheta[ int(nm*nf):].reshape((nu,nf))
    return (X,Theta)



def cofiCostFunc(pars,Y,R,nu,nm,nf,lam_par=0.0):
    X,Theta=reshapeParams(pars,nm,nu,nf)
    # const fuction
    cost=0.5*np.sum((X.dot(Theta.T)*R-Y)**2)
    # regularization
    cost+=(lam_par/2.)*np.sum(pars**2)    
    return cost



def cofiGrad(pars,Y,R,nu,nm,nf,lam_par=0.0):
    X,Theta=reshapeParams(pars,nm,nu,nf)
    tmp=X.dot(Theta.T)*R-Y
    # gradient
    gradX=tmp.dot(Theta)
    gradTheta=tmp.T.dot(X)
    # regularization
    gradX    +=lam_par*X
    gradTheta+=lam_par*Theta
    return flattenParams(gradX,gradTheta)



def checkGradient(pars,Y,R,nu,nm,nf,lam_par=0.0):
    print('Numerical Gradient \t cofiGrad \t\t Difference')
    eps=0.001
    npars=len(pars)
    epsvec=np.zeros(npars)
    grads=cofiGrad(pars,Y,R,nu,nm,nf,lam_par)
    for i in range(10):
        idx=np.random.randint(0,npars)
        epsvec[idx]=eps
        loss1=cofiCostFunc(pars-epsvec,Y,R,nu,nm,nf,lam_par)
        loss2=cofiCostFunc(pars+epsvec,Y,R,nu,nm,nf,lam_par)
        grad_num=(loss2-loss1)/(2*eps)
        epsvec[idx]=0
        print('%+0.15f \t %+0.15f \t %+0.15f'\
              %(grad_num,grads[idx],grad_num-grads[idx]))

#%% get data

data1=scipy.io.loadmat('data/ex8_movies.mat')
data1.keys()

R=data1.get('R')
Y=data1.get('Y')

nm,nu=Y.shape
# Y is 1682x943 containing ratings (1-5) of 1682 movies on 943 users
# a rating of 0 means the movie wasn't rated
# R is 1682x943 containing R(i,j) = 1 if user j gave a rating to movie i,
# i.e. R is redundant

#%% visualize data

print('Average rating for movie 1 (Toy Story):',Y[0,:].sum()/R[0,:].sum())

# "Visualize the ratings matrix"
plt.figure()
plt.imshow(Y)
plt.colorbar()
plt.ylabel('Movies (%d)'%nm,fontsize=20)
plt.xlabel('Users (%d)'%nu,fontsize=20)
plt.show()

#%% Get data for debugging

# Read in the movie params matrices
data2 = scipy.io.loadmat('data/ex8_movieParams.mat')
data2.keys()

X    =data2.get('X')
Theta=data2.get('Theta')
nu=data2.get('num_users'   )[0,0]
nm=data2.get('num_movies'  )[0,0]
nf=data2.get('num_features')[0,0]

## For now, reduce the data set size so that this runs faster
nu=4
nm=5
nf=3
X=X[:nm,:nf]
Theta=Theta[:nu,:nf]
Y=Y[:nm,:nu]
R=R[:nm,:nu]

#%% sanity check for the cost function

# "...run your cost function. You should expect to see an output of 22.22."
print('Cost with nu=4, nm=5, nf=3 is %0.2f.'\
      %cofiCostFunc(flattenParams(X,Theta),Y,R,nu,nm,nf))
    
# "...with lambda=1.5 you should expect to see an output of 31.34."
print('Cost with nu=4, nm=5, nf=3 (and lambda=1.5) is %0.2f.'\
      %cofiCostFunc(flattenParams(X,Theta),Y,R,nu,nm,nf,lam_par=1.5))

#%% sanity check for the gradient

print("Checking gradient with lambda = 0:")
checkGradient(flattenParams(X,Theta),Y,R,nu,nm,nf)
print("\nChecking gradient with lambda = 1.5:")
checkGradient(flattenParams(X,Theta),Y,R,nu,nm,nf,lam_par=1.5)










