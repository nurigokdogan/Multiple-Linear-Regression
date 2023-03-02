# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data_set = pd.read_csv('Companies.csv')
x = data_set.iloc[:,:-1].values
y = data_set.iloc[:,4].values 
imputer = SimpleImputer(missing_values=0.0 ,strategy="mean")
imputer = imputer.fit(x[:,0:3])
x[:,0:3] = imputer.transform(x[:,0:3])
label_encoder_x = LabelEncoder() 
x[:,3] = label_encoder_x.fit_transform(x[:,3])
onehot_encoder = ColumnTransformer([("Country" , OneHotEncoder(),[3])], remainder = 'passthrough')
x = nm.array(onehot_encoder.fit_transform(x))
x_train,x_test,y_train, y_test = train_test_split (x,y,test_size=0.2,random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
df = pd.DataFrame({'real values':y_test, 'predicted values':y_pred})
print(df)
plt.figure(figsize=(12,6))
plt.plot([0,1,2,3,4,5,6,7,8,9],y_pred, ':', color='r')
plt.plot([0,1,2,3,4,5,6,7,8,9],y_test, '-', color='y')
plt.show()