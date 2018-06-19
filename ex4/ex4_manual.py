#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 12:24:54 2018

@author: Anton Buyskikh
@brief: Neural network backpropagation.
"""

#%% libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import scipy.io

#%% functions

def h(X,theta):
    # hypothesis function
    # by default we use sigmoid
    return 1.0/(1.0+np.exp(-X.dot(theta)))



def sigmoid(z):
    return 1.0/(1+np.exp(-z))



def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))



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



def nnCostFunction(nn_params,\
                   layer0_size,\
                   layer1_size,\
                   layer2_size,
                   X,y,reg_param):
    """
    Computes loss using sum of square errors for a neural network
    using theta as the parameter vector for linear regression to fit 
    the data points in X and y with penalty reg_param.
    """
    m = len(y)    
    # Reshape nn_params back into neural network
    
    theta1=nn_params[:layer1_size*(layer0_size+1) ].reshape((layer1_size,layer0_size+1))
    theta2=nn_params[ layer1_size*(layer0_size+1):].reshape((layer2_size,layer1_size+1))
    
#    theta1 = nn_params[:(layer1_size * 
#			           (layer0_size + 1))].reshape((layer1_size,layer0_size +1))
#  
#    theta2 = nn_params[-((layer1_size + 1) * 
#                          layer2_size):].reshape((layer2_size,
#					                             layer1_size + 1))
   
    # Turn scalar y values into a matrix of binary 
    init_y = np.zeros((m,layer2_size)) # 5000 x 10
 
    for i in range(m):
        init_y[i][y[i]]=1

    # Add column of ones to X
    ones = np.ones((m,1)) 
    d = np.hstack((ones,X))# add column of ones
 
    # forward propogation with theta1 and theta2
    cost = [0]*m
    # Initalize gradient vector
    D1 = np.zeros_like(theta1)
    D2 = np.zeros_like(theta2)
    for i in range(m):
	
        a1 = d[i][:,None] # 401 x 1
        z2 = np.dot(theta1,a1) # 25 x 1 
        a2 = sigmoid(z2) # 25 x 1
        a2 = np.vstack((np.ones(1),a2)) # 26 x 1
        z3 = np.dot(theta2,a2) #10 x 1
        h = sigmoid(z3) # 10 x 1
        a3 = h # 10 x 1
        cost[i] = (np.sum((-init_y[i][:,None])*(np.log(h)) -
	              (1-init_y[i][:,None])*(np.log(1-h))))/m
	
        	# Calculate Gradient
        d3 = a3 - init_y[i][:,None]
        d2 = np.dot(theta2.T,d3)[1:]*(sigmoidGradient(z2))
	
        # Accumulate errors for gradient calculation
        D1 = D1 + np.dot(d2,a1.T) # 25 x 401 (matches theta0)
        D2 = D2 + np.dot(d3,a2.T) # 10 x 26 (matches theta1)

    # regularization
    reg = (reg_param/(2*m))*((np.sum(theta1[:,1:]**2)) + 
	      (np.sum(theta2[:,1:]**2)))
    
    # Compute final gradient with regularization
    grad1 = (1.0/m)*D1 + (reg_param/m)*theta1
    grad1[0] = grad1[0] - (reg_param/m)*theta1[0]
    
    grad2 = (1.0/m)*D2 + (reg_param/m)*theta2
    grad2[0] = grad2[0] - (reg_param/m)*theta2[0]
    
    # Append and unroll gradient
    grad = np.append(grad1,grad2).reshape(-1)
    final_cost = sum(cost) + reg

    return (final_cost, grad)

#%% PART I
# load data from Matlab files
data=scipy.io.loadmat('data/ex4data1.mat')
weights=scipy.io.loadmat('data/ex4weights.mat')
print('data    keys: ',data.keys())
print('weights keys: ',weights.keys())

#%% extract data

y=np.asarray(data['y'],dtype='int').ravel()
#y[y==10]=0
y = (y - 1) % 10

# Add constant for intercept
X=np.c_[np.ones((data['X'].shape[0],1)),np.asarray(data['X'])]
theta1_given=np.asarray(weights['Theta1'])
theta2_given=np.asarray(weights['Theta2'])

print('Shapes of variables:')
print('X: {} (with identity)'.format(X.shape))
print('y: {}'.format(y.shape))
print('Theta1: ',theta1_given.shape)
print('Theta2: ',theta2_given.shape)
print('\n')

#%% visualize data

sample=np.random.randint(0,X.shape[0],25)
fig,ax=displayData(X[sample,1:],y[sample])
fig.show()

#%%


layer0_size=400
layer1_size=25
layer2_size=10

# Unroll Parameters
#nn_params = np.append(theta1_given,theta2_given).reshape(-1)
nn_params=np.append(theta1_given.reshape([layer1_size*(layer0_size+1)]),\
                    theta2_given.reshape([layer2_size*(layer1_size+1)]))

print("Checking cost function without regularization...")
reg_param = 0.0
cost, g = nnCostFunction(nn_params,layer0_size,layer1_size,layer2_size,
		                     X[:,1:],y,reg_param)

# Test for correct cost
np.testing.assert_almost_equal(0.287629,cost,decimal=6, err_msg="Cost incorrect.")









#%%

















