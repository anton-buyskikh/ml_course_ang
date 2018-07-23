#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 08:55:56 2018

@author: Anton Buyskikh

@brief: Logistic regression with multiple features.
Polynomial features. Feature regularization.
"""

#%% libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

#%% functions

def plotData(X,y):
    pos=X[np.where(y==1)]
    neg=X[np.where(y==0)]
    fig,ax=plt.subplots()
    ax.plot(pos[:,0],pos[:,1],"k+",label='Admitted')
    ax.plot(neg[:,0],neg[:,1],"yo",label='Not Admitted')
    return (fig,ax)

#%% PART I
# load ex2data1.txt - logical regression with two parameters
    
data1=pd.read_csv('data/ex2data1.txt',names=['exam1','exam2','score'])
y=np.asarray(data1.score)
x=np.asarray(data1[["exam1","exam2"]])

#%% visualize data

fig,ax=plotData(x,y)
ax.legend()
fig.show()

#%% solution via sklearn

# main parameters to play with are degree and C

poly=PolynomialFeatures(degree=4,include_bias=False)
scaler=StandardScaler()

X=poly.fit_transform(x)
X=scaler.fit_transform(X)

regr_log=LogisticRegression(C=3)
regr_log.fit(X,y)

#%% plot the solution from the previous block

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh

# grid
h=0.5
x0_min,x0_max=x[:,0].min()-5*h,x[:,0].max()+5*h
x1_min,x1_max=x[:,1].min()-5*h,x[:,1].max()+5*h

# mesh
xx0,xx1=np.meshgrid(np.arange(x0_min,x0_max,h),np.arange(x1_min,x1_max,h))

# make poly features and scaling
XX=poly.fit_transform(np.c_[xx0.ravel(),xx1.ravel()])
XX=scaler.transform(XX)

# fit every mesh point
ZZ=regr_log.predict(XX)

# reshape to the origianl shape
ZZ=ZZ.reshape(xx0.shape)

#%% prediction and accuracy

# original score
x_pred=np.array([45,85]).reshape(1,-1)

# make poly features and scaling
X_pred=poly.fit_transform(x_pred)
X_pred=scaler.transform(X_pred)

# fitting result
y_pred=regr_log.predict_proba(X_pred)

print('For a student with scores (45,85), admission probability is %5.2f%%'\
      %(y_pred[0,1]*100))
print('Training Accuracy: %5.2f%%\n'%(regr_log.score(X,y)*100))

#%% visualize the decision boundary

fig,ax=plotData(x,y)
ax.pcolormesh(xx0,xx1,ZZ,cmap="RdYlBu")
ax.plot(x_pred[0,0],x_pred[0,1],'sk',label='Sample Student')
ax.legend()
fig.show()

#%% PART II





