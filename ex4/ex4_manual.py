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
    return g(X.dot(theta))



def g(z):
    return 1.0/(1+np.exp(-z))



def dg(z):
    return g(z)*(1-g(z))



def displayData(X,y):
    n1,n2=5,10
    fig,ax=plt.subplots(n1,n2,sharex=True,sharey=True)
    img_num=0
    for i in range(n1):
        for j in range(n2):
            # Convert column vector into 20x20 pixel matrix
            # You have to transpose to display correctly
            img=X[img_num,:].reshape(20,20).T
            ax[i][j].imshow(img,cmap='gray')
            ax[i][j].set_title(str(y[img_num]))
            ax[i][j].axis('off')
            img_num+=1
    return (fig,ax)



def getNNCostReg(nn_params,layer0_size,layer1_size,layer2_size,X,y,lam_par):
    """
    Forward the backward propagation on the neural network of 3 layers
    with regularization.
    TO_DO: generalize for any number of layers
    """
    # number of samples
    m=len(y)
    # create a vector for each value of y
    y_mat=np.zeros((m,layer2_size))
    for i in range(m):
        y_mat[i][y[i]]=1
    # since the matlab version of thetas start enumiration from 1, not from 0
    y_mat=np.roll(y_mat,-1,axis=1)
        
    # Reshape nn_params back into neural network
    theta1=nn_params[:layer1_size*(layer0_size+1) ].reshape((layer1_size,layer0_size+1))
    theta2=nn_params[ layer1_size*(layer0_size+1):].reshape((layer2_size,layer1_size+1))
 
    # forward propagation
    a1=np.c_[np.ones(( X.shape[0],1)),X ]
    z2=a1.dot(theta1.T)
    a2=g(z2)    
    a2=np.c_[np.ones((a2.shape[0],1)),a2]
    z3=a2.dot(theta2.T)
    a3=g(z3)
    
    # cost function with regularization
    J=(-y_mat*np.log(a3)-(1-y_mat)*(np.log(1-a3))).sum()/m
    J+=lam_par*((theta1[:,1:]**2).sum()+(theta2[:,1:]**2).sum())/2/m
    
    # gradient calculation
    # backward propagation: deltas
    d3=a3-y_mat
    d2=d3.dot(theta2)[:,1:]*dg(z2)

    # Deltas
    D1=d2.T.dot(a1)
    D2=d3.T.dot(a2)
    
    # gradients of each theta with regularization
    grad1=D1/m
    grad2=D2/m
    grad1[:,1:]+=lam_par/m*theta1[:,1:]
    grad2[:,1:]+=lam_par/m*theta2[:,1:]
    
    # reshape for the output
    grad=np.append(grad1,grad2).reshape(-1)

    return (J,grad)

#%% PART I
# load data from Matlab files
data=scipy.io.loadmat('data/ex4data1.mat')
weights=scipy.io.loadmat('data/ex4weights.mat')
print('data    keys: ',data.keys())
print('weights keys: ',weights.keys())

#%% extract data

y=np.asarray(data['y'],dtype='int').ravel()
y[y==10]=0

# Add constant for intercept
#X=np.c_[np.ones((data['X'].shape[0],1)),np.asarray(data['X'])]
X=np.asarray(data['X'])
theta1=np.asarray(weights['Theta1'])
theta2=np.asarray(weights['Theta2'])

print('Shapes of variables:')
print('X: {} (without identity)'.format(X.shape))
print('y: {}'.format(y.shape))
print('Theta1: ',theta1.shape)
print('Theta2: ',theta2.shape)
print('')

#%% visualize data

sample=np.random.randint(0,X.shape[0],50)
fig,ax=displayData(X[sample,:],y[sample])
fig.show()

#%% test forward propagation via the cost function

layer0_size=400
layer1_size=25
layer2_size=10

nn_params=np.append(theta1,theta2).reshape(-1)

print("Checking the cost function without regularization:")
lam_par=0.0
J,grad=getNNCostReg(nn_params,layer0_size,layer1_size,layer2_size,X,y,lam_par)
np.testing.assert_almost_equal(0.287629,J,decimal=6,\
                               err_msg="Cost function is NOT correct")
print('Cost function is correct\n')


print("Checking the cost function with regularization:")
lam_par=1.0
J,grad=getNNCostReg(nn_params,layer0_size,layer1_size,layer2_size,X,y,lam_par)
np.testing.assert_almost_equal(0.383770,J,decimal=6,\
                               err_msg="Cost function is NOT correct")
print('Cost function is correct\n')

#%%

















