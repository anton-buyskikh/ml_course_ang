#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:08:23 2018

@author: Anton Buyskikh
@brief: Support Vector Machines. Linear and Gaussian kernels.
...
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



def getGaussianKernel(x1,x2,sigma):
    # just an example of the kernel
    return np.exp(-(np.linalg.norm(x1-x2).sum()**2)/2/sigma**2)

#%% PART I
# load data

data1=scipy.io.loadmat('data/ex6data1.mat')
data1.keys()
X=data1.get('X')
y=data1.get('y').ravel()

#%% visualize data

fig,ax=plotData(X,y)
ax.legend(['Positive','Negative'])
fig.show()

#%% training SVM with a linear boundary

# C is the trade-off befween optimization and regularization of weights
# try changing C (=1/lambda) form 10^-2 to 10^2
C=1
svm=SVC(kernel='linear',C=C)
svm.fit(X,y)
weights=svm.coef_[0]
intercept=svm.intercept_[0]

#%% draw the linear boundary

xp=np.linspace(X.min(),X.max(),100)
yp=-(weights[0]*xp+intercept)/weights[1]

fig,ax=plotData(X,y)
ax.plot(xp,yp)
ax.legend(['Positive','Negative','Boundary for C='+str(C)])
fig.show()

#%% test the gaussian kernel

x1=np.array([1,2, 1])
x2=np.array([0,4,-1])
sigma=2
sim=getGaussianKernel(x1,x2,sigma)

print('Gaussian Kernel between x1=[1,2,1], x2=[0,4,-1], sigma={0} : {1}'.format(sigma,sim))
print('This value should be about 0.324652\n')

#%% load more complex data

data2=scipy.io.loadmat('data/ex6data2.mat')
data2.keys()
X=data2.get('X')
y=data2.get('y').ravel()

#%% visualize data

fig,ax=plotData(X,y)
ax.legend(['Positive','Negative'])
fig.show()

#%% training SVM with complex boundary

# here we use the gaussian kernel (rbf)
# C is the trade-off befween optimization and regularization of weights
# sigma is the "sharpness" of the contribution of each datapoint to the coutur plot
# try changing C and sigma
C=10.0
sigma=0.12
svm=SVC(kernel='rbf',C=C,gamma=1/2.0/sigma**2)
svm.fit(X,y)

#%% draw the complex boundary

x1=np.linspace(X[:,0].min(),X[:,0].max(),200)
x2=np.linspace(X[:,1].min(),X[:,1].max(),200)
x1,x2=np.meshgrid(x1,x2)
yp=svm.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape)

fig,ax=plotData(X,y)
plt.contour(x1,x2,yp)
ax.legend(['Positive','Negative'])
ax.set_title('C='+str(C)+', sigma='+str(sigma))
fig.show()











