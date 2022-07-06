

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.metrics import precision_score, recall_score,f1_score, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt

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
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.2, random_state=0)

model = Sequential()
model.add(Dense(200,input_dim = 3,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='mean_absolute_error',optimizer = Adam(lr = 0.001, beta_1=0.9, beta_2=0.999))
model.fit(x_train,y_train, validation_data = (x_val,y_val), verbose = 1, epochs = 200)
model_name = 'SNN_model'
model_json = model.to_json()
with open(f"{model_name}.json","w") as json_file:
    json_file.write(model_json)
model.save_weights(f"{model_name}.h5")

y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)
y_test_pred = model.predict(x_test)

train_accuracy = 100 - mean_absolute_percentage_error(y_train,y_train_pred)*100
val_accuracy = 100 - mean_absolute_percentage_error(y_val,y_val_pred)*100
test_accuracy = 100 - mean_absolute_percentage_error(y_test,y_test_pred)*100

print(train_accuracy)
print(val_accuracy)
print(test_accuracy)

# plot
fig, ax = plt.subplots()
ax.plot(y_train, y_train, linewidth=2.0)
ax.plot(y_train, y_train_pred,'o')
plt.title('Training')
plt.show()
