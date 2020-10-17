#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 19:07:55 2020

@author: chi
"""
import pandas as pd 
import numpy as np 
import scipy as sp
import matplotlib as mpl

from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import train_test_split

path ='/home/chi/Desktop/anaconda-navigator/Lab 1/spam.csv'
dataset_pd = pd.read_csv(path)
dataset_np = np.genfromtxt(path,delimiter=',')
X = dataset_np[:,0:len(dataset_np[0])-1]  
Y = dataset_np[:,len(dataset_np[0])-1]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.4,random_state=0)

#
    
clf=DecisionTreeClassifier(criterion='entropy')
clf=clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)
print("CART (Tree Prediction) Accuracy: {}".format(sum(Y_pred==Y_test)/len(Y_pred)))
print("CART (Tree Prediction) Accuracy by calling metrics: ",metrics.accuracy_score(Y_test,Y_pred)) 

#

scores=cross_val_score(clf,X,Y,cv=5)
print("scores={} \n final score={} \n".format(scores,scores.mean()))
print("\n")

#
clf=SVC()
#Fit SVM Classifier
clf.fit(X_train,Y_train)
#Predict testset
Y_pred=clf.predict(X_test)
#Evaluate performance of the model
print("SVM Accuracy: ",metrics.accuracy_score(Y_test,Y_pred))
print("\n")#Evaluate a score by cross-validation
scores=cross_val_score(clf,X,Y,cv=5)
print("scores={}\n final score={}\n".format(scores,scores.mean()))
print("\n")

#
 
#Fit Random Forest Classifier
rdf=RandomForestClassifier()
rdf.fit(X_train,Y_train)
#Predict testset
Y_pred=rdf.predict(X_test)
#Evaluate performance of the model
print("RDF: ",metrics.accuracy_score(Y_test,Y_pred))
print("\n")
#Evalute a score by cross validation
scores=cross_val_score(rdf,X,Y,cv=5)
print("scores={}\n final score={}\n".format(scores,scores.mean()))
print("\n")

#

#Fit Logistic Regression Classifier
lr=LogisticRegression()
lr.fit(X_train,Y_train)
#Predict testset
Y_pred=lr.predict(X_test)
#Evaluate performance of the model
print("LR: ", metrics.accuracy_score(Y_test,Y_pred))
#Evalute a score by cross validation
scores=cross_val_score(lr,X,Y,cv=5)
print("scores={} \n final score=() \n".format(scores,scores.mean()))
print("\n")