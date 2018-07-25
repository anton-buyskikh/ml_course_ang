#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 18:43:31 2018

@author: Anton Buyskikh
@brief: Neural network classifier (TensorFlow)
"""

#%% libraries

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

import tensorflow as tf

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
X=np.asarray(data['X'])

#%% visualize data

sample=np.random.randint(0,X.shape[0],50)
fig,ax=displayData(X[sample,:],y[sample])
fig.show()

#%% define the basic model

model=tf.keras.models.Sequential([\
      tf.keras.layers.Flatten(),\
      tf.keras.layers.Dense(25,activation=tf.nn.relu),\
      tf.keras.layers.Dropout(0.2),\
      tf.keras.layers.Dense(10,activation=tf.nn.softmax)])
    
model.compile(optimizer='adam',\
              loss='sparse_categorical_crossentropy',\
              metrics=['accuracy'])

#%% fit the model

model.fit(X,y,epochs=10)
accuracy=model.evaluate(X,y)

print('Training Accuracy: %5.2f%%\n'%(accuracy[1]*100))

#%% Visualize hidden layers

weights=model.get_weights()

for i in range(int(len(weights)/2)):
    fig,ax=displayData(weights[2*i].T)
    fig.show()

