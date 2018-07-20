#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:32:59 2018

@author: Anton Buyskikh
@brief: Recommender Systems. Collaborative filtering.
"""
#%% libraries

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import random

#%% functions

def flattenParams(X,Theta):
    return np.concatenate((X.flatten(),Theta.flatten()))



def reshapeParams(XTheta,nm,nu,nf):
    # X:     (nm x nf)
    # Theta: (nu x nf)
    assert XTheta.size==int(nm*nf+nu*nf)
    X    =XTheta[:int(nm*nf) ].reshape((nm,nf))
    Theta=XTheta[ int(nm*nf):].reshape((nu,nf))
    return (X,Theta)



def cofiCostFunc(pars,Y,R,nu,nm,nf,lam_par=0.0):
    X,Theta=reshapeParams(pars,nm,nu,nf)
    # const fuction
    # NB: only sum elements that have R==1
    cost=0.5*np.sum((X.dot(Theta.T)*R-Y)**2)
    # regularization
    cost+=(lam_par/2.)*np.sum(pars**2)    
    return cost



def cofiGrad(pars,Y,R,nu,nm,nf,lam_par=0.0):
    X,Theta=reshapeParams(pars,nm,nu,nf)
    # NB: only sum elements that have R==1
    tmp=X.dot(Theta.T)*R-Y
    # gradient
    gradX=tmp.dot(Theta)
    gradTheta=tmp.T.dot(X)
    # regularization
    gradX    +=lam_par*X
    gradTheta+=lam_par*Theta
    return flattenParams(gradX,gradTheta)



def checkGradient(pars,Y,R,nu,nm,nf,lam_par=0.0):
    print('Numerical Gradient \t cofiGrad \t\t Difference')
    eps=0.001
    npars=len(pars)
    epsvec=np.zeros(npars)
    grads=cofiGrad(pars,Y,R,nu,nm,nf,lam_par)
    for i in range(10):
        idx=np.random.randint(0,npars)
        epsvec[idx]=eps
        loss1=cofiCostFunc(pars-epsvec,Y,R,nu,nm,nf,lam_par)
        loss2=cofiCostFunc(pars+epsvec,Y,R,nu,nm,nf,lam_par)
        grad_num=(loss2-loss1)/(2*eps)
        epsvec[idx]=0
        print('%+0.15f \t %+0.15f \t %+0.15f'\
              %(grad_num,grads[idx],grad_num-grads[idx]))



def normalizeRatings(Y,R):
    # The mean is only counting movies that were rated
    Ymean=np.sum(Y,axis=1)/np.sum(R,axis=1)
    Ymean=Ymean.reshape((-1,1))
    return (Y-Ymean,Ymean)

#%% get data

data1=scipy.io.loadmat('data/ex8_movies.mat')
data1.keys()

R=data1.get('R')
Y=data1.get('Y')

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

#%% Get data for debugging

# Read in the movie params matrices
data2 = scipy.io.loadmat('data/ex8_movieParams.mat')
data2.keys()

X    =data2.get('X')
Theta=data2.get('Theta')
nu=data2.get('num_users'   )[0,0]
nm=data2.get('num_movies'  )[0,0]
nf=data2.get('num_features')[0,0]

#%% sanity check for the cost function

## For now, reduce the data set size so that this runs faster
nu1=4
nm1=5
nf1=3
X1=X[:nm1,:nf1]
Theta1=Theta[:nu1,:nf1]
Y1=Y[:nm1,:nu1]
R1=R[:nm1,:nu1]

# "...run your cost function. You should expect to see an output of 22.22."
print('Cost with nu=4, nm=5, nf=3 is %0.2f.'\
      %cofiCostFunc(flattenParams(X1,Theta1),Y1,R1,nu1,nm1,nf1))
    
# "...with lambda=1.5 you should expect to see an output of 31.34."
print('Cost with nu=4, nm=5, nf=3 (and lambda=1.5) is %0.2f.'\
      %cofiCostFunc(flattenParams(X1,Theta1),Y1,R1,nu1,nm1,nf1,lam_par=1.5))

#%% sanity check for the gradient

print("Checking gradient with lambda = 0:")
checkGradient(flattenParams(X,Theta),Y,R,nu,nm,nf)
print("\nChecking gradient with lambda = 1.5:")
checkGradient(flattenParams(X,Theta),Y,R,nu,nm,nf,lam_par=1.5)

#%% load movie Ids

movieList=pd.read_table('data/movie_ids.txt',encoding='latin-1',names=['Info'])
movieList=pd.DataFrame(movieList.Info.str.split(' ',1).tolist(),columns=['Index','Movie'])
movieList=movieList.drop(['Index'],axis=1)

#%% add my ratings
     
Y_my=np.zeros((nm,1))
Y_my[0]  =4
Y_my[97] =2
Y_my[6]  =3
Y_my[11] =5
Y_my[53] =4
Y_my[63] =5
Y_my[65] =3
Y_my[68] =5
Y_my[182]=4
Y_my[225]=5
Y_my[354]=5

# add my ratings to Y matrix, and the relevant columnt to R matrix
R_my=Y_my>0
Y=np.hstack((Y,Y_my))
R=np.hstack((R,R_my))
nm,nu=Y.shape

#%% normalize data (this is optional)

Ynorm,Ymean=normalizeRatings(Y,R)

#%% generate random initial parameters, Theta and X

X_opt     =np.random.rand(nm,nf)
Theta_opt =np.random.rand(nu,nf)

#%% train the model with fmin_cg

XTheta_opt=flattenParams(X_opt,Theta_opt)

result=scipy.optimize.minimize(cofiCostFunc,\
                               XTheta_opt,\
                               args=(Y,R,nu,nm,nf,10.0),\
                               method='CG',\
                               tol=1e-7,\
                               jac=cofiGrad,\
                               options={'maxiter':100,'disp':True})
X_opt,Theta_opt=reshapeParams(result.x,nm,nu,nf)

# NB:
# 1. one can use bounds or constrains to enforce rating to be in the interval
#    from 0 to 5
# 2. also it's possible via constraints to enforce that the provided ratings
#    are preserved
# 3. such things as normalization and regularization require futher 
#    investigation as well

#%% calculate the preditions

# If the normalized data was used add back in the mean movie ratings
#pred_all=X_opt.dot(Theta_opt.T)+Ymean
pred_all=X_opt.dot(Theta_opt.T)

# optionally one can also restore all initial ratings
i1,i2=np.where(R)
pred_all[i1,i2]=Y[i1,i2]

# extract my prediction
pred_my=pred_all[:,-1]

#%% visualize the predicted data

plt.figure()
plt.imshow(pred_all)
plt.colorbar()
plt.ylabel('Movies (%d)'%nm,fontsize=20)
plt.xlabel('Users (%d)'%nu,fontsize=20)
plt.show()

#%% Sort my predictions from highest to lowest

ind_pred_sort=np.argsort(pred_my)[::-1]

print("Top recommendations for you:")
for i in range(30):
    print('Predicting rating %0.1f for movie %s.'\
          %(pred_my[ind_pred_sort[i]],movieList.Movie[ind_pred_sort[i]]))
    
print("\nOriginal ratings provided:")
for i in range(len(Y_my)):
    if Y_my[i]>0:
        print('Rated %d for movie %s.'%(Y_my[i],movieList.Movie[i]))






