import numpy as np
import pandas as pd

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler


data=pd.read_csv('Prediction-Of-type-2-Diabetes/diabetes.csv')

scaler = MinMaxScaler()
data['Age'] = scaler.fit_transform(data[['Age']])
data['Glucose'] = scaler.fit_transform(data[['Glucose']])
data['BloodPressure'] = scaler.fit_transform(data[['BloodPressure']])
data['BMI'] = scaler.fit_transform(data[['BMI']])
data['SkinThickness'] = scaler.fit_transform(data[['SkinThickness']])
data['Pregnancies'] = scaler.fit_transform(data[['Pregnancies']])
data['Insulin']=scaler.fit_transform(data[['Insulin']])


X=data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y=data[['Outcome']]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
lr=LogisticRegression()
lr.fit(X_train,y_train.values.ravel())



y_pred=lr.predict(X_test)


pickle.dump(lr,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))