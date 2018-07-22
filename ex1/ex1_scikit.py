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
from sklearn.preprocessing import StandardScaler

#%% PART I
# load ex1data1.txt - linear regression with one parameter
    
data1=pd.read_csv("data/ex1data1.txt",names=["X","y"])
x=data1.X.values[:,None]
y=data1.y.values

poly=PolynomialFeatures(1)
X=poly.fit_transform(x)

#%% use sklearn

# pick models
regr_gd=SGDRegressor(fit_intercept=False,alpha=0.00001,max_iter=10000)
regr_lr=LinearRegression(fit_intercept=False)

# feed data
regr_gd.fit(X,y)
regr_lr.fit(X,y)

#%% plot the solution via the Gradient Decent

ind=x.argsort(axis=0).flatten()

fig,ax=plt.subplots() # create empty figure
plt.plot(x,y,'rx',label='Training data')
plt.plot(x[ind],X[ind,:].dot(regr_lr.coef_),'-k',label='lin. reg. (sklearn)')
plt.plot(x[ind],X[ind,:].dot(regr_gd.coef_),'-b',label='stoch. grad. descent (sklearn)')
ax.set_xlabel("Population of City in 10,000s")
ax.set_ylabel("Profit in $10,000s")
ax.legend()
plt.show()

#%% PART II
# load dataset2
 
data2=pd.read_csv("data/ex1data2.txt",names=["area","nbed","price"])
y=data2.price.values
x=np.hstack((data2.area.values[:,None],data2.nbed.values[:,None]))

poly=PolynomialFeatures(1)
X=poly.fit_transform(x)

#%% scale data

scaler=StandardScaler()
X[:,1:]=scaler.fit_transform(X[:,1:])

#%% train sklearn models

# pick models
regr_gd=SGDRegressor(fit_intercept=False,alpha=0.0001,max_iter=100000)
regr_lr=LinearRegression(fit_intercept=False)

# feed data
regr_gd.fit(X,y)
regr_lr.fit(X,y)

#%% prediction

# initial parameters
predict=np.array([1650,3]).reshape(1,-1)

# add features
predict=poly.fit_transform(predict)

# rescale
predict[:,1:]=scaler.transform(predict[:,1:])

print('Predicted price of a 1650 sq-ft, 3 br house (using sklearn lr): \n',
       regr_lr.predict(predict))
print('Predicted price of a 1650 sq-ft, 3 br house (using sklearn gd): \n',
       regr_gd.predict(predict))


