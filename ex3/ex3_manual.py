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


#%% PART I
# load data and weights from Matlab files
data=scipy.io.loadmat('data/ex3data1.mat')
weights=scipy.io.loadmat('data/ex3weights.mat')

print('data    keys: ',data.keys())
print('weights keys: ',weights.keys())

#%% extract data

y=data['y']
# Add constant for intercept
X=np.c_[np.ones((data['X'].shape[0],1)),data['X']]

theta1,theta2=weights['Theta1'],weights['Theta2']

print('Shapes of variables:')
print('X: {} (with identity)'.format(X.shape))
print('y: {}'.format(y.shape))
print('theta1: {}'.format(theta1.shape))
print('theta2: {}'.format(theta2.shape))
print('\n')

#%% visualize data

sample=np.random.choice(X.shape[0],20)
plt.imshow(X[sample,1:].reshape(-1,20).T)
plt.axis('off');

#%% create and test cost and gradient function



#%% solution via the optimization algorithm



#%% plot the solution from the previous block



#%% prediction and accuracy



#%% PART II

    


#%% visualize data



#%% create and test cost and gradient function



#%% solution via the optimization algorithm



#%% plot the solution from the previous block



