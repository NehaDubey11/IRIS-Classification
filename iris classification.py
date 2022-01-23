# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 16:03:57 2022

@author: 91639
"""

from sklearn import datasets
iris=datasets.load_iris()
X=iris.data
Y=iris.target

from sklearn.model_selection import train_test_split

x_test,x_train,y_test,y_train=\
    train_test_split(X,Y,test_size=0.3,random_state=1234,stratify=Y)
    
    #train the model

from sklearn.svm import SVC
svc=SVC(kernel="rbf",gamma=1)
svc.fit(x_train,y_train)
y_predict=svc.predict(x_test)

from sklearn.metrics import confusion_matrix
cm_rbf1=confusion_matrix(y_test,y_predict)
score=svc.score(x_test,y_test)

svc=SVC(kernel="rbf",gamma=10)
svc.fit(x_train,y_train)
y_predict=svc.predict(x_test)
cm_rbf10=confusion_matrix(y_test,y_predict)
score=svc.score(x_test,y_test)


svc=SVC(kernel="linear")
svc.fit(x_train,y_train)
y_predict=svc.predict(x_test)
cm_linear=confusion_matrix(y_test,y_predict)
score=svc.score(x_test,y_test)

svc=SVC(kernel="poly")
svc.fit(x_train,y_train)
y_predict=svc.predict(x_test)
cm_poly=confusion_matrix(y_test,y_predict)
score=svc.score(x_test,y_test)

svc=SVC(kernel="sigmoid")
svc.fit(x_train,y_train)
y_predict=svc.predict(x_test)
cm_sig=confusion_matrix(y_test,y_predict)
score=svc.score(x_test,y_test)
