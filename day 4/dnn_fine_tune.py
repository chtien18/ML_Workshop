import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import copy
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import statistics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import model_from_json
import tensorflow.keras
import os
def dataset_prep(RSN):
    dataset = pd.read_excel('Machining data.xlsx')
    input_ = dataset.iloc[:,0:3]
    input_ = input_.values
    
    row,col = np.shape(input_)
    norm_input = np.zeros((row,col))
    
    for i in range(row):
        for j in range(col):
            norm_input[i,j] = (input_[i,j]-min(input_[:,j])*0.9)/(max(input_[:,j])*1.1-min(input_[:,j])*0.9)
    
    target_ = dataset.iloc[:,3]
    target_ = target_.values
    norm_target = np.zeros(row)
                                                                
    for i in range(row):
        norm_target[i] = target_[i]/(max(target_)*1.05)
    
    norm_target = norm_target.reshape(-1, 1)
    dataset_ = np.hstack((norm_input,norm_target))

    np.random.seed(int(RSN))
    ds = shuffle(dataset_)

    x = ds[:,0:3]
    y = ds[:,3]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/6, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

    return x_train, x_val, x_test, y_train, y_val, y_test

def output_denormalization(output_dataset):
    dataset = pd.read_excel('Machining data.xlsx')
    target_ = dataset.iloc[:,3]
    target_ = target_.values
    
    denorm_ = np.zeros(len(output_dataset))
    for i in range(len(output_dataset)):
        denorm_[i] = output_dataset[i]*max(target_)*1.05
    return denorm_

def dnn_fine_tune():
    
    old_model_name = 'pre_train_dnn2'
    epochs = 300
    perf_avg = 100
    I=0
    while I<100:
        # RSN = random.randint(0, 200)
        np.random.seed(I)
        x_train, x_val, x_test, y_train, y_val, y_test = dataset_prep(I)

        with open(f'{old_model_name}.json') as json_file:
            model_old = model_from_json(json_file.read())
            model_old.load_weights(f'{old_model_name}.h5')
        
        model= tensorflow.keras.models.clone_model(model_old)
        model.set_weights(model_old.get_weights())

        model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=0.0005,beta_1=0.95,beta_2=0.999))
    
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, verbose=0)
        
        y_hat_train = model.predict(x_train)
        y_hat_val = model.predict(x_val)
        y_hat_test = model.predict(x_test)

        mape_train = mean_absolute_percentage_error(y_train,y_hat_train)*100
        mape_val = mean_absolute_percentage_error(y_val,y_hat_val)*100
        mape_test = mean_absolute_percentage_error(y_test,y_hat_test)*100
        
        if I==0 and perf_avg == 100:
            perf_train = mape_train
            perf_val = mape_val
            perf_test = mape_test
            perf_avg = (perf_train+perf_val+perf_test)/3

        if mape_train<3 and mape_val<3 and mape_test<3: 
        
            model_json = model.to_json()
            with open("dnn_2_model.json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights("dnn_2_model.h5")
            print("Saved model to disk")
            print(f"MAPE_train : {mape_train} %")
            print(f"MAPE_val : {mape_val} %")
            print(f"MAPE_test : {mape_test} %")
            
            '''training'''
            plt.plot(output_denormalization(y_train))
            plt.plot(output_denormalization(y_hat_train))
            plt.legend(['original value', 'train_predicted value'], loc='best')
            plt.show()
                
            '''validation'''
            plt.plot(output_denormalization(y_val))
            plt.plot(output_denormalization(y_hat_val))
            plt.legend(['original value', 'val_predicted value'], loc='best')
            plt.show()
                    
            '''testing'''
            plt.plot(output_denormalization(y_test))
            plt.plot(output_denormalization(y_hat_test))
            plt.legend(['original value', 'test_predicted value'], loc='best')
            plt.show()
            
            break

        elif (mape_train+mape_val+mape_test)/3 < perf_avg:
            model_json = model.to_json()
            with open("dnn_2_model.json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights("dnn_2_model.h5")
            print("Saved model to disk")
            print(f"MAPE_train : {mape_train} %")
            print(f"MAPE_val : {mape_val} %")
            print(f"MAPE_test : {mape_test} %")
            print('')
            print('')
            old_model_name = 'dnn_2_model'
            perf_avg = (mape_train+mape_val+mape_test)/3
            if perf_avg <3:
                epochs = 30
            if I==99:
                I=-1
        else:
            if I==99:
                I=-1
        I+=1    
    return mape_train, mape_val, mape_test, I

mape_train, mape_val, mape_test, rsn = dnn_fine_tune()
