# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import pickle

dataset = pd.read_excel('Machining data.xlsx')

input_ = dataset.iloc[:,0:3]
input_ = input_.values
scaler = MinMaxScaler()
scaler.fit(input_)
input_ = scaler.transform(input_)

target_ = dataset.iloc[:,3]
target_ = target_.values

target_1 = np.zeros(len(target_))
for i in range(len(target_1)):
    target_1[i] = target_[i]/max(target_)
    
target_1 = target_1.reshape(-1,1)


dataset_ = np.hstack((input_,target_1))

np.random.seed(100)
ds = shuffle(dataset_)

x = ds[:,0:3]
y = ds[:,3]
y=y.astype('float64')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#model = SVR(kernel ='poly', C=10, gamma = 'auto', degree=3, epsilon=0.1, coef0=0.01)
#model = RandomForestRegressor()
model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=80, n_jobs=-1)

model.fit(x_train,y_train)

filename = 'SVR_poly_model.sav'
pickle.dump(model, open(filename,'wb'))

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_accuracy = 100-mean_absolute_percentage_error(y_train, y_train_pred)*100
test_accuracy = 100-mean_absolute_percentage_error(y_test, y_test_pred)*100
print(train_accuracy)
print(test_accuracy)