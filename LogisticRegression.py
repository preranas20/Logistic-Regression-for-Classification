#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 20:33:52 2017
@author: preranasingh
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
import matplotlib.lines as mlines
from sklearn import datasets


def sigmoid_func(theta, x):
    return float(1) / (1 + np.exp(-np.dot(x,theta)))


def derivative_Jtheta(h_theta_x,X_input,Y_input):
    derivative_cost=np.zeros((99,3))
    for i,h_theta_x_i in enumerate(h_theta_x):
        diff_cost=h_theta_x_i-(Y_input[i])
        derivative_cost[i]=diff_cost*X_input[i]
    derived_cost=(np.sum(derivative_cost,axis=0))/99 
    return derived_cost


def calculate_Jtheta(theta,X_input,Y_input):
    temp_cost=np.zeros((100,1))
    h_theta_x=sigmoid_func(theta,X_input)
    Y_input = np.squeeze(Y_input)
    for i,h_theta_x_i in enumerate(h_theta_x):
        e1=Y_input[i]*np.log(h_theta_x_i)
        e2=(1-(Y_input[i]))*np.log(1-h_theta_x_i)
        temp_cost[i]=-e1-e2
    J_theta=(np.sum(temp_cost,axis=0))/99
    return J_theta     
   
def pred_values(theta,X_test):
   
    predicted_prob = sigmoid_func(theta,X_test)#0.654
    if predicted_prob >= 0.5:
        predicted_y=1
    else:
        predicted_y=0
    return predicted_y    
            
        

#Loading the data and scaling it
iris = datasets.load_iris()
X_petal_length=iris.data[50:150,2:3]
petal_length_num=X_petal_length-(np.amin(X_petal_length))
petal_length_den=np.amax(X_petal_length)-np.amin(X_petal_length)
X_scale_length=petal_length_num/petal_length_den

X_petal_width=iris.data[50:150,3:4]
petal_width_num=X_petal_width-(np.amin(X_petal_width))
petal_width_den=np.amax(X_petal_width)-np.amin(X_petal_width)
X_scale_width=petal_width_num/petal_width_den
X=np.column_stack((np.ones(100),X_scale_length,X_scale_width))

Y=iris.target[0:100]


error=0

for i in range(0,100):
    X_test=X[i]
    X_train=[x for idx,x in enumerate(X) if idx!=i]
    Y_test=Y[i]
    Y_train=[y for idx2,y in enumerate(Y) if idx2!=i]
    theta=np.zeros((3,1))
    cost=np.zeros((99,1))

    Y_train = np.squeeze(Y_train)
    
    h_theta_x=sigmoid_func(theta,X_train)

    
    for i,h_theta_x_i in enumerate(h_theta_x):
         e1=Y_train[i]*np.log(h_theta_x_i)
         e2=(1-(Y_train[i]))*np.log(1-h_theta_x_i)
         cost[i]=-e1-e2

    J_theta=(np.sum(cost,axis=0))/99

    cost_iter = []
    cost_iter.append([0, cost])
    learning_rate=0.001

    #minimizing cost until convergence
    for itr in range(450):
        JTheta_derived=np.reshape(derivative_Jtheta(h_theta_x, X_train,Y_train),(3,1))
        theta[0,0]=theta[0,0]-(learning_rate *(JTheta_derived[0,0]))
        theta[1,0]=theta[1,0]-(learning_rate *(JTheta_derived[1,0]))
        theta[2,0]=theta[2,0]-(learning_rate *(JTheta_derived[2,0]))
        cost2 = calculate_Jtheta(theta,X_train,Y_train)
        cost_iter.append([i, cost2])
        
       
    #Testing on test data    
    predicted_y = pred_values(theta,X_test) 
    result=predicted_y - Y_test.astype(np.int64)
    
    if result==0:
        error=error+0
    else:
        error=error+1
        
    
avg_error=error/100    