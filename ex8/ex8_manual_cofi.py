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



def cofiCostFunc(pars,Y,R,nu,nm,nf,lam_par=0.):
    X,Theta=reshapeParams(pars,nm,nu,nf)
    # const fuction
    cost=0.5*np.sum((X.dot(Theta.T)*R-Y)**2)
    # regularization
    cost+=(lam_par/2.)*np.sum(pars**2)    
    return cost

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


















