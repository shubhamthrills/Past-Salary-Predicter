# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:40:36 2019

@author: shubham
"""
#Past Salary Predict Model
sal1=float(input("New Employee past salary (as per employee):"))
working_exp=input("Enter your Working Experision:")
working_exp=float(working_exp)
#print(working_exp)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('info.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
#Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""
#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
#Fitting Random Forest Regression to the 'info' dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)
#Predicting a new result
y_pred = regressor.predict(working_exp)
print("Predicted Salary is: ")
print(y_pred)
#diff=y_pred-sal1
pdiff=sal1-y_pred
if(pdiff>0):
    print("Employee is laying and difference is:")
    print(pdiff)
else:
    print("Great Employee is Saying Truth and Difference is:")
    print(pdiff)   
#Visualising the Graph (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Predicting the salary Model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()