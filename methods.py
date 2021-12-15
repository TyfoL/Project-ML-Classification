# -*- coding: utf-8 -*-
# author@ JI Xueyao
import math
import numpy
import random
import pandas as pd

def sigmoid(x,theta):
    z = 0
    for i in range (4):
        z = z + x[i]*theta[i]
    sum = 1/(1.0+math.exp(-z))
    return sum
    
def derivative(X,Y,theta,size,j): # j represents dimension
    sum = 0.0
    for i in range(size):
        sum = sum + (sigmoid(X[i],theta)-Y[i])*X[i][j]
    return (sum/size)

def loss(X,Y,theta,size):
    sum = 0.0
    for i in range(size):
        h = sigmoid(X[i],theta)
        sum = sum + (h-Y[i])*(h-Y[i])
    return (sum/2*size)
            
def gradient(X,Y,theta,size,rate,count):
    for i in range(count):
        t = []
        for j in range(4):
            h = theta[j] - rate*derivative(X,Y,theta,size,j)
            t.append(h)
        for j in range(4):
            theta[j] = t[j]
    
def logistic_regression(data_path, num_gradient_descent=30000, rate=0.002):
    data = pd.read_csv(data_path,header=None)
    X = data[[0,1,2,3]] # the first four values are eigenvalues
    Y = data[[4]]       # the last value is discriminant value
    size = data.shape[0]
    theta = numpy.zeros((4)) # 4 represents the number of individuals instead of the index (which starts from 0)
    gradient(X.values.tolist(),Y[4].tolist(),theta,size,rate,num_gradient_descent)
      
    X, Y = X.values.tolist(), Y.values.tolist()
    b = 0
    for i in range(1000): 
        z = 0
        j = random.randint(0,1371)
        for k in range (4):
            z = z + X[j][k]*theta[k]
        sum = 1/(1.0+math.exp(-z))
        if sum > 0.5 :
            if Y[j][0] == 1:
                b = b + 1
        elif Y[j][0] == 0:
            b = b + 1
    return b/1000
