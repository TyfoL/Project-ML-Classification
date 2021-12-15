# -*- coding: utf-8 -*-
# author@ JI Xueyao
import math
import numpy
import random
import pandas as pd
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

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


def Svm_acc(X_train, X_test, y_train, y_test):
    model = svm.SVC(kernel='linear', gamma='scale')
    model.fit(X_train,y_train)
    y_pred_svm = model.predict(X_test)
    svm_accuracy = accuracy_score(y_pred_svm,y_test)
    return svm_accuracy


def DecisionTree_acc(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(criterion='entropy',spliltter='best',random_state=0)
    model.fit(X_train,y_train)
    y_pred_dt = model.predict(X_test)
    dt_accuracy = accuracy_score(y_pred_dt,y_test)
    return dt_accuracy


def Kmeans_acc(X_train, X_test, y_train, y_test):
    model = KMeans(n_clusters=2, random_state=0)
    model.fit(X_train,y_train)
    y_pred_km = model.predict(X_test)
    km_accuracy = accuracy_score(y_pred_km,y_test)
    return km_accuracy
