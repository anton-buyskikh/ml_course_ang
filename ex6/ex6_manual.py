#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:08:23 2018

@author: Anton Buyskikh
@brief: Support Vector Machines.
"""

#%% libraries

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#%% functions

def plotData(X,y):
    pos=X[np.where(y==1)]
    neg=X[np.where(y==0)]
    fig,ax=plt.subplots()
    ax.plot(pos[:,0],pos[:,1],"k+",neg[:,0],neg[:,1],"yo")
    return (fig,ax)

#%% PART I
# load data

data=scipy.io.loadmat('data/ex6data1.mat')
data.keys()
X=data.get('X')
y=data.get('y').ravel()

#%% visualize data

fig,ax=plotData(X,y)
ax.legend(['Positive','Negative'])
fig.show()

#%% training linear SVM

# C is the trade-off befween optimization and regularization of weights
# try changing C (=1/lambda) form 10^-2 to 10^2
C=1
svm=SVC(kernel='linear',C=C)
svm.fit(X,y)
weights=svm.coef_[0]
intercept=svm.intercept_[0]

#%% draw the SVM boundary

xp=np.linspace(X.min(),X.max(),100)
yp=-(weights[0]*xp+intercept)/weights[1]

fig,ax=plotData(X,y)
ax.plot(xp,yp)
ax.legend(['Positive','Negative','Boundary for C='+str(C)])
fig.show()

#%%













