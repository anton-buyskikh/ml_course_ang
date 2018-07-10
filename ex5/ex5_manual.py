#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 12:10:34 2018

@author: Anton Buyskikh
@brief: Regulirized linear regression. High bias vs high variance models.
Learning curve for models with polynomial features.
"""

#%% libraries

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#%% functions

def h(X,theta):
    # linear regression hypothesis
    return X.dot(theta)



def getCostReg(theta,X,y,lam_par):
    m,n=X.shape
    J=((h(X,theta)-y)**2).mean()/2+lam_par*(theta[1:]**2).sum()/2/m
    return J



def getGradReg(theta,X,y,lam_par):
    m=X.shape[0]
    gradJ=(h(X,theta)-y).dot(X)/m
    gradJ[1:]+=lam_par/m*theta[1:]
    return gradJ



def trainLinearReg(theta_init,X,y,lam_par,disp_opt=True):
    res=scipy.optimize.minimize(getCostReg,\
                                theta_init,\
                                args=(X,y,lam_par),\
                                method='Newton-CG',\
                                tol=1e-5,\
                                jac=getGradReg,\
                                options={'maxiter':1000,'disp':disp_opt})    
    return res



def learningCurve(X_train,y_train,X_cv,y_cv,lam_par,niter=100):
    # total length of the training set
    m=y_train.size
    error_train_mean=np.zeros(m)
    error_train_std =np.zeros(m)
    error_cv_mean   =np.zeros(m)
    error_cv_std    =np.zeros(m)
    
    # One chooses niter sample sets of the length i and obrain error for
    # the training set as well as for the cross-validation set.
    # Then one averages over all iterations.
    
    for i in range(m):
        tmp_train=np.zeros(niter)
        tmp_cv   =np.zeros(niter)
        
        for iter in range(niter):
            sample=np.sort(random.sample(range(m),i+1))
            theta_init=np.random.uniform(low=-0.1,high=0.1,size=X_train.shape[1])
            res=trainLinearReg(theta_init,X_train[sample,:],y_train[sample],\
                               lam_par,disp_opt=False)
            tmp_train[iter]=getCostReg(res.x,X_train[sample,:],y_train[sample],0.0)
            tmp_cv   [iter]=getCostReg(res.x,X_cv             ,y_cv           ,0.0)
        
        error_train_mean[i]=tmp_train.mean()
        error_train_std[i] =tmp_train.std()
        error_cv_mean[i]   =tmp_cv.mean()
        error_cv_std[i]    =tmp_cv.std()
    
    return (error_train_mean,error_train_std,error_cv_mean,error_cv_std)



def normalizeFeatures(X,**kwargs):
    if 'mean' in kwargs:
        mean=kwargs['mean']
    else:
        mean=np.mean(X,axis=0)
        
    if 'std' in kwargs:
        std=kwargs['std']
    else:
        std =np.std(X,axis=0)
        
    return (np.divide((X-mean),std),mean,std)



def validationCurve(X_train,y_train,X_cv,y_cv):
    lam_vec=np.array([0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]).reshape((-1,1))
    error_train=np.zeros(len(lam_vec))
    error_cv   =np.zeros(len(lam_vec))
    
    for i in range(len(lam_vec)):
        theta_init=np.random.uniform(low=-0.1,high=0.1,size=X_train.shape[1])
        res=trainLinearReg(theta_init,X_train,y_train,lam_vec[i],disp_opt=False)
        error_train[i]=((X_train.dot(res.x)-y_train)**2).mean()/2.0
        error_cv[i]   =((X_cv.dot(res.x)   -y_cv   )**2).mean()/2.0
        
    return (lam_vec,error_train,error_cv)
	

#%% PART I: underfitting with a linear features
# load data

data=scipy.io.loadmat('data/ex5data1.mat')
data.keys()

#%% extract data

# load y-data
y_train=data.get('y').ravel()
y_test =data.get('ytest').ravel()
y_cv   =data.get('yval').ravel()

# load only linear features
poly=PolynomialFeatures(1)
X_train=poly.fit_transform(data.get('X'    ))
X_test =poly.fit_transform(data.get('Xtest'))
X_cv   =poly.fit_transform(data.get('Xval' ))

# print shapes
print('X_train:',X_train.shape)
print('y_train:',y_train.shape)
print('X_test:' ,X_test.shape)
print('y_test:' ,y_test.shape)
print('X_cv:'   ,X_cv.shape)
print('y_cv:'   ,y_cv.shape)

#%% visualize data

fig,ax=plt.subplots()
ax.plot(X_train[:,1],y_train,'xk',label='train')
ax.plot(X_test[:,1] ,y_test ,'ob',label='test')
ax.plot(X_cv[:,1]   ,y_cv   ,'sr',label='cross-validation')
ax.set_xlabel('Change in water level (x)')
ax.set_ylabel('Water flowing out of the dam (y)')
ax.legend() 
fig.show()

#%% define initial theta and find its cost function with its gradient

lam_par=0.0
theta_init=np.ones(X_train.shape[1])
J=getCostReg(theta_init,X_train,y_train,0.0)
gradJ=getGradReg(theta_init,X_train,y_train,0.0)
print('Cost functiona at the initial point: ',J)
print('Gradient at the initial point: ',gradJ)

#%% find optimal theta using the training set

res=trainLinearReg(theta_init,X_train,y_train,lam_par)
print('theta optimal (manual): ',res.x)
print('Cost funstion minimum (manual): ',res.fun)

#%% comparison with LinearRegression in Scikit-learn

regr=LinearRegression(fit_intercept=False)
regr.fit(X_train,y_train)
print('theta optimal (Scikit-learn): ',regr.coef_)
print('Cost funstion minimum (Scikit-learn): ',getCostReg(regr.coef_,X_train,y_train,lam_par))
# one should see the same results

#%% plot the result with only linear features

fig,ax=plt.subplots()
ax.plot(X_train[:,1],y_train,'xk',label='train')
ax.plot(np.linspace(-50,40),(res.x[0]+(res.x[1]*np.linspace(-50,40))),label='linear fit')
ax.set_xlabel('Change in water level (x)')
ax.set_ylabel('Water flowing out of the dam (y)')
ax.legend() 
fig.show()

# this is clearly an underfitting model, hence we should observe an evidence
# for high bias

#%% investigate bias-variance crosover of the model

error_train_mean,error_train_std,error_cv_mean,error_cv_std=\
    learningCurve(X_train,y_train,X_cv,y_cv,0.0,niter=200)

#%% plot the learning curve for the linaer model

fig,ax=plt.subplots()
ax.errorbar(range(1,len(y_train)+1),error_train_mean,error_train_std,\
            label='Training error',capsize=2)
ax.errorbar(range(1,len(y_train)+1),error_cv_mean   ,error_cv_std   ,\
            label='Validation error',capsize=2)
ax.set_title('Learning curve for linear regression')
ax.set_xlabel('Number of training examples')
ax.set_ylabel('Error')
ax.legend()
ax.set_ylim(ymin=0)
fig.show()

# As it was expected we see a large error as the size of the samles increases.
# Hence the model has the high bias (underfitting).

#%% PART II: overfitting with high degree polynomials

# Consider polynomial features
poly=PolynomialFeatures(8)
X_train_poly=poly.fit_transform(data.get('X'    ))
X_test_poly =poly.fit_transform(data.get('Xtest'))
X_cv_poly   =poly.fit_transform(data.get('Xval' ))

# normalize them
X_train_poly[:,1:],X_mean,X_std=normalizeFeatures(X_train_poly[:,1:])
X_test_poly[:,1:],_,_          =normalizeFeatures(X_test_poly[:,1:],mean=X_mean,std=X_std)
X_cv_poly[:,1:],_,_            =normalizeFeatures(X_cv_poly[:,1:],mean=X_mean,std=X_std)

#%% find optimal theta using the training set

# try 0 (overfitting), 1 (just right) and 100 (badly fitter)
lam_par=1.0
theta_init=np.random.uniform(low=-0.1,high=0.1,size=X_train_poly.shape[1])
res=trainLinearReg(theta_init,X_train_poly,y_train,lam_par)
print('theta optimal (manual): ',res.x)
print('Cost funstion minimum (manual): ',res.fun)


#%% plot the result with polynomial features

x_fit=np.linspace(-70,60).reshape(-1,1)
X_fit=poly.fit_transform(x_fit)
X_fit[:,1:],_,_=normalizeFeatures(X_fit[:,1:],mean=X_mean,std=X_std)
y_fit=X_fit.dot(res.x)


fig,ax=plt.subplots()
ax.plot(X_train[:,1],y_train,'xk',label='train')
ax.plot(x_fit,y_fit,label='poly-'+str(poly.degree)+' fit')
ax.set_xlabel('Change in water level (x)')
ax.set_ylabel('Water flowing out of the dam (y)')
ax.legend() 
fig.show()

# This is an overfitting model at lambda~0, all training data points have small
# errors. Now let's the learning curves.

#%% investigate bias-variance crosover of the model with polynomial features

lam_par=0.01
error_train_poly_mean,error_train_poly_std,error_cv_poly_mean,error_cv_poly_std=\
    learningCurve(X_train_poly,y_train,X_cv_poly,y_cv,lam_par,niter=100)

#%% plot the learning curve for the polynomial model

fig,ax=plt.subplots()
ax.errorbar(range(1,len(y_train)+1),error_train_poly_mean,\
            error_train_poly_std,label='Training error',capsize=2)
ax.errorbar(range(1,len(y_train)+1),error_cv_poly_mean   ,\
            error_cv_poly_std   ,label='Validation error',capsize=2)
ax.set_title('Learning curve for the linear regression of poly-'+str(poly.degree)+' features')
ax.set_xlabel('Number of training examples')
ax.set_ylabel('Error')
ax.legend()
ax.grid()
fig.show()

# As it was expected we see a small error of the training set as the size of
# the samles increases, and the cross-validation error is significantly larger,
# (here it's an order of magnitude larger).
# Hence the model has the high variance (overfitting), however it can be fixed
# with the regularization parameter.
# SPOILER: you can try lambda~3, which will give "just right" fitting

#%% Validation curves

lam_vec,error_train,error_cv=validationCurve(X_train_poly,y_train,\
                                             X_cv_poly,y_cv)

#%% plot validation curves

fig,ax=plt.subplots()
ax.plot(lam_vec,error_train,'-xk',label='train curve')
ax.plot(lam_vec,error_cv   ,'-ob',label='cross-val. curve')
ax.set_xlabel('lambda')
ax.set_ylabel('error')
ax.legend() 
fig.show()

# one can observe that the lowest error of the cross-validation set is at lam~3

#%% test error using the optimal lambda from above

lam_vec,error_train,error_test=validationCurve(X_train_poly,y_train,\
                                             X_test_poly,y_test)

fig,ax=plt.subplots()
ax.plot(lam_vec,error_train,'-xk',label='train curve')
ax.plot(lam_vec,error_test ,'-ob',label='test curve')
ax.set_xlabel('lambda')
ax.set_ylabel('error')
ax.legend() 
fig.show()

# Error at lam~3 is indeed the smallest and very close to the one in ex5.pdf