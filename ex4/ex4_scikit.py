#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:08:26 2018

@author: Anton Buyskikh
@brief: Neural network classifier.
"""

#%% libraries

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures

#%% functions

def displayData(X,y=None):
    # assume there always will be 5 rows
    n1=5
    n2=int(X.shape[0]/n1)
    # assume each image is square
    img_size=int(X.shape[1]**0.5)
    # plotting
    fig,ax=plt.subplots(n1,n2,sharex=True,sharey=True)
    img_num=0
    for i in range(n1):
        for j in range(n2):
            # Convert column vector into 20x20 pixel matrix
            # You have to transpose to display correctly
            img=X[img_num,:].reshape(img_size,img_size).T
            ax[i][j].imshow(img,cmap='gray')
            if y is not None:
                ax[i][j].set_title(str(y[img_num]))
            ax[i][j].axis('off')
            img_num+=1
    return (fig,ax)

#%% PART I
# load data from Matlab files
data=scipy.io.loadmat('data/ex4data1.mat')
print('data    keys: ',data.keys())

#%% extract data

y=np.asarray(data['y']).ravel()
y[y==10]=0 # for some reason they call 0 as 10
x=np.asarray(data['X'])

#%% visualize data

sample=np.random.randint(0,x.shape[0],50)
fig,ax=displayData(x[sample,:],y[sample])
fig.show()

#%% add poly features if necessary

poly=PolynomialFeatures(degree=1,include_bias=False)
X=poly.fit_transform(x)

#%% solution via sklearn

regr_mlp=MLPClassifier(alpha=0.001,
                       max_iter=2000,
                       hidden_layer_sizes=(25,),
                       activation='logistic',
                       solver='lbfgs')
regr_mlp.fit(X,y)

print('Training Accuracy: %5.2f%%\n'%(regr_mlp.score(X,y)*100))

#%% Visualize hidden layers

for i in range(regr_mlp.n_layers_-1):
    fig,ax=displayData(regr_mlp.coefs_[i].T)
    fig.show()


