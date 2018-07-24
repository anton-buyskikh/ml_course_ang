#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 17:36:56 2018

@author: Anton Buyskikh

@brief: Logistic regression with multi-class classification.
Neural network method.
"""

#%% libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import scipy.io

from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

from sklearn.preprocessing import PolynomialFeatures

#%% functions

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

#%% PART I
# load data from Matlab files
data=scipy.io.loadmat('data/ex3data1.mat')
print('data keys: ',data.keys())

#%% extract data

y=np.asarray(data['y']).ravel()
y[y==10]=0
x=np.asarray(data['X'])

#%% visualize data

sample=np.random.randint(0,x.shape[0],25)
fig,ax=displayData(x[sample,:],y[sample])
fig.show()

#%% add poly features if necessary

poly=PolynomialFeatures(degree=1,include_bias=False)
X=poly.fit_transform(x)

#%% solution via sklearn

regr_log=LogisticRegression(C=10,solver='liblinear')
regr_log.fit(X,y)

# accuracy
print('Training Accuracy: %5.2f%%\n'%(regr_log.score(X,y)*100))

