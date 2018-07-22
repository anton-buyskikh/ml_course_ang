#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 10:47:50 2018

@brief: Linear regression with one and multiple features.
Features normalization.
"""

#%% libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit

from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict

#%% PART I
# load ex1data1.txt - linear regression with one parameter
    
data1=pd.read_csv("data/ex1data1.txt",names=["X","y"])
x=data1.X.values[:,None]
poly=PolynomialFeatures(1)
X=poly.fit_transform(x)
y=data1.y.values

#%% use sklearn

# pick models
regr_sgd=SGDRegressor(fit_intercept=False,alpha=0.00001,max_iter=10000)
regr_lr =LinearRegression(fit_intercept=False)

# feed data
regr_sgd.fit(X[:,0:],y)
regr_lr.fit( X[:,0:],y)

#%% plot the solution via the Gradient Decent

fig,ax=plt.subplots() # create empty figure
plt.plot(x,y,'rx',label='Training data')
plt.plot(x,X.dot(regr_lr.coef_ ),label='lin. reg. (sklearn)')
plt.plot(x,X.dot(regr_sgd.coef_),label='stoch. grad. descent (sklearn)')
ax.set_xlabel("Population of City in 10,000s")
ax.set_ylabel("Profit in $10,000s")
ax.legend()
plt.show()

#%% PART II
