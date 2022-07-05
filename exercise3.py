import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

dataset = pd.read_excel('Mall_Customers.xlsx')

ds = dataset.iloc[:,0:4]
ds = ds.values #pick the number only

scaler = MinMaxScaler()
scaler.fit(ds)
ds = scaler.transform(ds)

np.random.seed(10)
dataset = shuffle(dataset)

x = ds[:,0:3]
y = ds[:,3]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.25,random_state=0) #25% of training data for validation

