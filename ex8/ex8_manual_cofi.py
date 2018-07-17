#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:32:59 2018

@author: Anton Buyskikh
@brief: Recommender Systems.
...
"""
#%% libraries

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random

#%% functions


#%% get data

data=scipy.io.loadmat('data/ex8_movies.mat')
data.keys()

R=data.get('R')
Y=data.get('Y')

nm,nu=Y.shape
# Y is 1682x943 containing ratings (1-5) of 1682 movies on 943 users
# a rating of 0 means the movie wasn't rated
# R is 1682x943 containing R(i,j) = 1 if user j gave a rating to movie i,
# i.e. R is redundant

#%% visualize data

print('Average rating for movie 1 (Toy Story):',Y[0,:].sum()/R[0,:].sum())

# "Visualize the ratings matrix"
plt.figure()
plt.imshow(Y)
plt.colorbar()
plt.ylabel('Movies (%d)'%nm,fontsize=20)
plt.xlabel('Users (%d)'%nu,fontsize=20)
plt.show()

#%%













