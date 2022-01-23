import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

loan_data=pd.read_csv('01Exercise1.csv')

loan_data_prep=loan_data.copy()
print(loan_data_prep.isnull().sum(axis=0))

loan_prep=loan_data_prep.dropna()
loan_prep=loan_prep.drop(['gender'],axis=1)

#create dummy variables

loan_prep=pd.get_dummies(loan_prep,drop_first=True)

from sklearn.preprocessing import StandardScaler
scaler_=StandardScaler()
loan_prep['income']=scaler_.fit_transform(loan_prep[['income']])
loan_prep['loanamt']=scaler_.fit_transform(loan_prep[['loanamt']])

#create x and y dataframes
Y=loan_prep[['status_Y']]
X=loan_prep.drop(['status_Y'],axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=\
    train_test_split(X,Y,test_size=0.3,random_state=1234,stratify=Y)
    
    
#build support vector  classifier
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_predict=svc.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)
score=svc.score(x_test,y_test)











