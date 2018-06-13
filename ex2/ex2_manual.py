#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 09:34:37 2018

@author: Anton Buyskikh

@brief: Logistic regression with multiple features.
...
"""

#%% libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit

#from scipy.optimize import minimize
import scipy

from sklearn.preprocessing import PolynomialFeatures

#%% functions

def plotData(X,y):
    pos=X[np.where(y==1)]
    neg=X[np.where(y==0)]
    fig,ax=plt.subplots()
    ax.plot(pos[:,0],pos[:,1],"k+",neg[:,0],neg[:,1],"yo")
    return (fig,ax)



def plotDataNew(data, label_x, label_y, label_pos, label_neg, axes=None):
    # Get indexes for class 0 and class 1
    neg = data[:,2] == 0
    pos = data[:,2] == 1
    
    # If no specific axes object has been passed, get the current axes.
    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:,0], data[pos][:,1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:,0], data[neg][:,1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon= True, fancybox = True);



def h(X,theta):
    # hypothesis function
    # by default we use sigmoid
    return 1.0/(1.0+np.exp(-X@theta))



def getCostAndGrad(theta,X,y):
    # Returns the cost function and its gradient
    # see Sec. 1.2.2
    m,n=X.shape
    J=np.mean(-y*np.log(h(X,theta))-(1-y)*(np.log(1-h(X,theta))))
    grad=(h(X,theta)-y)@X/m
    return (J,grad)



def getCostAndGradReg(theta,X,y,lam_par):
    # Returns the regularized cost function and its gradient
    # see Sec. 2.3
    J,grad=getCostAndGrad(theta,X,y)
    m,n=X.shape
    J+=lam_par/2*(theta[1:]**2).mean()
    grad[1:]+=lam_par/m*theta[1:]
    return (J,grad)



def getCostReg(theta,X,y,lam_par):
    J,grad=getCostAndGradReg(theta,X,y,lam_par)
    return J



def getGradReg(theta,X,y,lam_par):
    J,grad=getCostAndGradReg(theta,X,y,lam_par)
    return grad



def predict(X,theta):
    return np.where(h(X,theta)>0.5,1,0)



#def mapFeatureVector(X1,X2):
#    degree=6
#    output_feature_vec = np.ones(len(X1))[:,None]
#    for i in range(1,degree+1):
#        for j in range(i+1):
#            new_feature=np.array(X1**(i-j)*X2**j)[:,None]
#            output_feature_vec = np.hstack((output_feature_vec,new_feature))
#    return output_feature_vec



def plotDecisionBoundary(theta,X,y):
    fig,ax=plotData(X[:,1:],y)
    u=np.linspace(-1,1.5,50)
    v=np.linspace(-1,1.5,50)
    z=np.zeros((len(u),len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i][j] = np.dot(poly.fit_transform([np.array([u[i],v[j]])]),theta)
    ax.contour(u,v,z,levels=[0.])
    return (fig,ax)




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
print('Train Accuracy: ',np.mean(predict(X,theta_opt)==y)*100, '\n')

#%% PART II
# load ex2data2.txt - logical regression with two parameters
    
data2=pd.read_csv('data/ex2data2.txt',names=['test1','test2','acceptance'])
y=np.asarray(data2.acceptance)
X=np.hstack((np.ones_like(y)[:,None],np.asarray(data2[["test1","test2"]])))

#%% visualize data

fig,ax=plotData(X[:,1:],y)
ax.legend(['Good','Bad'])
fig.show()

#%% create polynomial features

poly=PolynomialFeatures(6)
XX=poly.fit_transform(data2[["test1","test2"]])
XX.shape

#%% create and test cost and gradient function

theta_init=np.zeros(XX.shape[1])
cost,grad=getCostAndGradReg(theta_init,XX,y,0.0)
print('Cost at theta_init=[0 0 0]: \n', cost)
print('Gradient at theta_init=[0 0 0]: \n',grad, '\n')

#%% solution via the optimization algorithm

lam_par=1.0

#res=scipy.optimize.minimize(getCostAndGradReg,\
#                            theta_init,\
#                            args=(XX,y,lam_par),\
#                            method='Newton-CG',\
#                            jac=True,\
#                            options={'maxiter':400,'disp':True})

res=scipy.optimize.minimize(getCostReg,\
                            theta_init,\
                            args=(XX,y,lam_par),\
                            method=None,\
                            jac=getGradReg,\
                            options={'maxiter':3000,'disp':True})

theta_opt=res.x
print('theta_opt: ',theta_opt)
print('Cost at theta_opt: ',res.fun, '\n')

#%% plot the solution from the previous block

x1_min,x1_max=X[:,1].min(),X[:,1].max(),
x2_min,x2_max=X[:,2].min(),X[:,2].max(),
xx1,xx2=np.meshgrid(np.linspace(x1_min,x1_max),np.linspace(x2_min,x2_max))
sol=h(poly.fit_transform(np.c_[xx1.ravel(),xx2.ravel()]),theta_opt)
sol=sol.reshape(xx1.shape)

accuracy=100*np.mean(predict(XX,theta_opt)==y)

fig,ax=plotData(X[:,1:],y)
ax.contour(xx1,xx2,sol,[0.5], linewidths=1, colors='g');       
ax.set_xlabel('Microchip test 1')
ax.set_ylabel('Microchip test 2')
ax.set_title('Lambda='+str(lam_par)+', accuracy='+str(accuracy))
ax.legend(['Pass', 'Fail','Decision Boundary'])
fig.show()



