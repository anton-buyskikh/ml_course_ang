#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 09:34:37 2018

@author: Anton Buyskikh

@brief: Logistic regression with multiple features.
Polynomial features. Feature regularization.
"""

#%% libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import scipy
from sklearn.preprocessing import PolynomialFeatures

#%% functions

def plotData(X,y):
    pos=X[np.where(y==1)]
    neg=X[np.where(y==0)]
    fig,ax=plt.subplots()
    ax.plot(pos[:,0],pos[:,1],"k+",neg[:,0],neg[:,1],"yo")
    return (fig,ax)



def h(X,theta):
    # hypothesis function
    # by default we use sigmoid
    return 1.0/(1.0+np.exp(-X@theta))



def getCostAndGrad(theta,X,y):
    # Returns the cost function and its gradient
    # see Sec. 1.2.2
    m,n=X.shape
    J=(-y*np.log(h(X,theta))-(1-y)*(np.log(1-h(X,theta)))).mean()
    grad=(h(X,theta)-y)@X/m
    return (J,grad)



def getCostAndGradReg(theta,X,y,lam_par):
    # Returns the regularized cost function and its gradient
    # see Sec. 2.3
    J,grad=getCostAndGrad(theta,X,y)
    m,n=X.shape
    J+=lam_par*(theta[1:]**2).sum()/2/m
    grad[1:]+=lam_par/m*theta[1:]
    return (J,grad)



def predict(X,theta,threshold=0.5):
    return np.where(h(X,theta)>threshold,1,0)

#%% PART I
# load ex2data1.txt - logical regression with two parameters
    
data1=pd.read_csv('data/ex2data1.txt',names=['exam1','exam2','score'])
y=np.asarray(data1.score)
X=np.hstack((np.ones_like(y)[:,None],np.asarray(data1[["exam1","exam2"]])))

#%% visualize data

fig,ax=plotData(X[:,1:],y)
ax.legend(['Admitted','Not admitted'])
fig.show()

#%% create and test cost and gradient function

theta_init=np.zeros(3)
cost,grad=getCostAndGrad(theta_init,X,y)
print('Cost at theta_init=[0 0 0]: \n', cost)
print('Gradient at theta_init=[0 0 0]: \n',grad, '\n')

#%% solution via the optimization algorithm

res=scipy.optimize.minimize(getCostAndGrad,
                            theta_init,
                            args=(X,y),
                            method='Newton-CG',
                            jac=True,
                            options={'maxiter':400,'disp':True})

theta_opt=res.x
print('theta_opt: ',theta_opt)
print('Cost at theta_opt: ',res.fun, '\n')

#%% plot the solution from the previous block

#NB: there should be a nicer way!!!
x1=np.linspace(30,100,61)
x2=(-theta_opt[0]-theta_opt[1]*x1)/theta_opt[2]
fig,ax=plotData(X[:,1:],y)
ax.plot(x1,x2,'-r')
ax.legend(['Admitted','Not admitted','Decision boundary'])
fig.show()

#%% prediction and accuracy

print('For a student with scores (45,85), admission probability is',
      h(np.array([1,45,85]),theta_opt))

# Accuracy on the training set with the optimal parameters
print('Train Accuracy: ',np.mean(predict(X,theta_opt,0.5)==y)*100, '\n')

#%% PART II
# load ex2data2.txt - logical regression with two parameters
    
data2=pd.read_csv('data/ex2data2.txt',names=['test1','test2','acceptance'])
y=np.asarray(data2.acceptance)
poly=PolynomialFeatures(6)
X=poly.fit_transform(data2[["test1","test2"]])
X.shape

#%% visualize data

fig,ax=plotData(X[:,1:3],y)
ax.legend(['Good','Bad'])
fig.show()

#%% create and test cost and gradient function

theta_init=np.zeros(X.shape[1])
cost,grad=getCostAndGradReg(theta_init,X,y,0.0)
print('Cost at theta_init=[0 0 0]: \n', cost)
print('Gradient at theta_init=[0 0 0]: \n',grad, '\n')

#%% solution via the optimization algorithm

# Try 0,1,100
lam_par=1.0

res=scipy.optimize.minimize(getCostAndGradReg,\
                            theta_init,\
                            args=(X,y,lam_par),\
                            method='Newton-CG',\
                            jac=True,\
                            options={'maxiter':3000,'disp':True})

theta_opt=res.x
print('theta_opt: ',theta_opt)
print('Cost at theta_opt: ',res.fun, '\n')

#%% plot the solution from the previous block

# create the mesh grid
x1_min,x1_max=X[:,1].min(),X[:,1].max(),
x2_min,x2_max=X[:,2].min(),X[:,2].max(),
xx1,xx2=np.meshgrid(np.linspace(x1_min,x1_max,100),np.linspace(x2_min,x2_max,100))

# solve for every grid point
sol=h(poly.fit_transform(np.c_[xx1.ravel(),xx2.ravel()]),theta_opt)
sol=sol.reshape(xx1.shape)
accuracy=100*np.mean(predict(X,theta_opt,0.5)==y)

fig,ax=plotData(X[:,1:],y)
ax.contour(xx1,xx2,sol,[0.5], linewidths=1, colors='g');       
ax.set_xlabel('Microchip test 1')
ax.set_ylabel('Microchip test 2')
ax.set_title('Lambda='+str(lam_par)+', accuracy='+str(accuracy))
ax.legend(['Pass', 'Fail','Decision Boundary'])
fig.show()



