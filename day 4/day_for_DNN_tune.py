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
N=5
Max_iter =2

def myround(x, prec=0, base=1):
    return round(base * round(float(x)/base),prec)

def myround_2(x, prec=2, base=0.01):
    return round(base * round(float(x)/base),prec)

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

def dnn2_training(D1,D2,DO1,DO2,RSN,mape_record):
    x_train, x_val, x_test, y_train, y_val, y_test = dataset_prep(RSN)
    
    model = Sequential()
    model.add(Dense(D1, input_dim=3, activation='relu'))
    model.add(Dropout(DO1))
    model.add(Dense(D2, activation ='relu'))
    model.add(Dropout(DO2))
    model.add(Dense(1, activation='sigmoid'))
        
    model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=0.0005,beta_1=0.95,beta_2=0.999))
    model.fit(x_train,y_train,validation_data=(x_val,y_val),verbose=0,epochs=100)

    y_hat_train = model.predict(x_train)
    y_hat_val = model.predict(x_val)
    y_hat_test = model.predict(x_test)

    mape_train = mean_absolute_percentage_error(y_train,y_hat_train)*100
    mape_val = mean_absolute_percentage_error(y_val,y_hat_val)*100
    mape_test = mean_absolute_percentage_error(y_test,y_hat_test)*100

    fitness_ = (mape_train+mape_val+mape_test)/3
    
    if fitness_<min(mape_record):
        mape_record.append(fitness_)
        model_json = model.to_json()
        with open(f"pre_train_{fitness_}.json", "w") as json_file:
            json_file.write(model_json)
            model.save_weights(f"pre_train_{fitness_}.h5")

    return fitness_, mape_record

def model_MAPE(mape_record, Gbest):    # To see the MAPE of the best model
    for i in range(len(mape_record)):
        if mape_record[i] == min(mape_record):
            
            with open(f'pre_train_{mape_record[i]}.json') as json_file:
                model_old = model_from_json(json_file.read())
                model_old.load_weights(f'pre_train_{mape_record[i]}.h5')
            
            model_= tensorflow.keras.models.clone_model(model_old)
            model_.set_weights(model_old.get_weights())
    
    old_file_name = str(min(mape_record))
    
    os.rename(f'pre_train_{old_file_name}.json','pre_train_dnn2.json')
    os.rename(f'pre_train_{old_file_name}.h5','pre_train_dnn2.h5')

    RSN = Gbest[-1]
    x_train, x_val, x_test, y_train, y_val, y_test = dataset_prep(RSN)
    
    y_h_train = model_.predict(x_train)
    y_h_val = model_.predict(x_val)
    y_h_test = model_.predict(x_test)
    
    mape_train = mean_absolute_percentage_error(y_train,y_h_train)*100
    mape_val = mean_absolute_percentage_error(y_val,y_h_val)*100
    mape_test = mean_absolute_percentage_error(y_test,y_h_test)*100
    
    model_mape = np.zeros(3)
    model_mape[0] = mape_train
    model_mape[1] = mape_val
    model_mape[2] = mape_test
    
    return model_mape

'''  ---------------  PSO -------------------'''

def position_initialization(N):
    initial_position = np.zeros((N,5))
    D_min = 1
    D_max = 200
    DO_min = 0
    DO_max = 0.5
    RSN_min = 0
    RSN_max = 100

    for i in range(N):
        initial_position[i,0] = myround(D_min + random.uniform(0, 1)*(D_max-D_min))
        initial_position[i,1] = myround(D_min + random.uniform(0, 1)*(D_max-D_min))
        initial_position[i,2] = myround_2(DO_min + random.uniform(0, 1)*(DO_max-DO_min))
        initial_position[i,3] = myround_2(DO_min + random.uniform(0, 1)*(DO_max-DO_min))
        initial_position[i,4] = myround(RSN_min + random.uniform(0, 1)*(RSN_max-RSN_min))
    return initial_position

def find_index(F):
    i = 0
    while i < len(F):
        if F[i] == min(F):
            break
            i = i
        else:
            i+=1
    return i

def boundary_handling(X):
    N = len(X)
    D_min = 1
    D_max = 200
    DO_min = 0
    DO_max = 0.5
    RSN_min = 0
    RSN_max = 100

    for i in range(N):
        if X[i,0] < D_min:
            X[i,0] = myround(D_min + random.uniform(0, 1)*(D_max-D_min))
        elif X[i,0] > D_max:
            X[i,0] = myround(D_min + random.uniform(0, 1)*(D_max-D_min))
        
        if X[i,1] < D_min:
            X[i,1] = myround(D_min + random.uniform(0, 1)*(D_max-D_min))
        elif X[i,1] > D_max:
            X[i,1] = myround(D_min + random.uniform(0, 1)*(D_max-D_min))
        
        if X[i,2] < DO_min:
            X[i,2] = myround_2(DO_min + random.uniform(0, 1)*(DO_max-DO_min))
        elif X[i,2] > DO_max:
            X[i,2] = myround_2(DO_min + random.uniform(0, 1)*(DO_max-DO_min))
        
        if X[i,3] < DO_min:
            X[i,3] = myround_2(DO_min + random.uniform(0, 1)*(DO_max-DO_min))
        elif X[i,3] > DO_max:
            X[i,3] = myround_2(DO_min + random.uniform(0, 1)*(DO_max-DO_min))
        
        if X[i,4] < RSN_min:
            X[i,4] = myround(RSN_min + random.uniform(0, 1)*(RSN_max-RSN_min))
        elif X[i,4] > RSN_max:
            X[i,4] = myround(RSN_min + random.uniform(0, 1)*(RSN_max-RSN_min))
    return X

def pso_opt(N,Max_iter):
    mape_record = []
    mape_record.append(1000)
    
    n_var = 5
    fitnest_history_0 = []
    fitnest_history_1 = []
    
    X = position_initialization(N)   # Initial positions
    Xnew = np.zeros((N,n_var))          
    V = 0.1*X                        # Initial velocity
    Vnew = np.zeros((N,n_var))
    
    c1 = 2
    c2 = 2
    
    I = 0

    while I < Max_iter:
        
        fitness_new = np.zeros(len(X))
        for i in range(len(X)):
            D1 = int(X[i,0])
            D2 = int(X[i,1])
            DO1 = X[i,2]
            DO2 = X[i,3]
            RSN = int(X[i,4])
            
            fitness_new[i], mape_record = dnn2_training(D1,D2,DO1,DO2,RSN,mape_record)
        
        if I ==0:
            X = X
            V = V
            Pbest = copy.deepcopy(X)
            fitness_Pbest = copy.deepcopy(fitness_new)
            idx = find_index(fitness_Pbest)
            Gbest = Pbest[idx,:]
        
        else:
            for i in range(N):   # memory saving
                if fitness_new[i] < fitness_Pbest[i]:
                    Pbest[i,:] = copy.deepcopy(X[i,:])
                    fitness_Pbest[i] = copy.deepcopy(fitness_new[i])
                else:
                    Pbest[i,:] = copy.deepcopy(Pbest[i,:])
                    fitness_Pbest[i] = copy.deepcopy(fitness_Pbest[i])
        
        idx = find_index(fitness_Pbest)
            
        Gbest = Pbest[idx,:]
        
        fitnest_history_0.append(min(fitness_Pbest))
        fitnest_history_1.append(np.mean(fitness_Pbest))
        print(f'Iteration {I}')
        print(f'minimum fitness: {min(fitness_Pbest)}')
        print(f'average fitness: {np.mean(fitness_Pbest)}')

        if abs(np.mean(fitness_Pbest)-min(fitness_Pbest)) < 0.01: #convergent criterion
            break

        r1 = np.zeros((N,n_var))
        r2 = np.zeros((N,n_var))
        for i in range(N):
            for j in range(n_var):
                r1[i,j] = random.uniform(0,1)
                r2[i,j] = random.uniform(0,1)
                
        Gbest = Gbest.reshape(1, -1)

        for i in range(N):
            for j in range(n_var):
                w_max = 0.9
                w_min = 0.4
                w = (w_max-w_min)*(Max_iter-I)/Max_iter + w_min
                Vnew[i,j] = w*V[i,j] + c1*r1[i,j]*(Pbest[i,j]-X[i,j]) + c2*r2[i,j]*(Gbest[0,j]-X[i,j])
                Xnew[i,j] = X[i,j] + Vnew[i,j]
        
        X = copy.deepcopy(Xnew)
        V = copy.deepcopy(Vnew)
        
        X = boundary_handling(X)
        
        for i in range(len(X)):
            X[i,0] = myround(X[i,0])
            X[i,1] = myround(X[i,1])
            X[i,2] = myround_2(X[i,2])
            X[i,3] = myround_2(X[i,3])
            X[i,4] = myround(X[i,4])
            
        I+=1
        
    fitness_new = np.zeros(len(X))
    for i in range(len(X)):
            D1 = int(X[i,0])
            D2 = int(X[i,1])
            DO1 = X[i,2]
            DO2 = X[i,3]
            RSN = int(X[i,4])
            
            fitness_new[i], mape_record = dnn2_training(D1,D2,DO1,DO2,RSN,mape_record)
            
    for i in range(N):
        if fitness_new[i] < fitness_Pbest[i]:
            Pbest[i,:] = copy.deepcopy(X[i,:])
            fitness_Pbest[i] = copy.deepcopy(fitness_new[i])
        else:
            Pbest[i,:] = copy.deepcopy(Pbest[i,:])
            fitness_Pbest[i] = copy.deepcopy(fitness_Pbest[i])
            
    idx = find_index(fitness_Pbest)
                    
    Gbest = Pbest[idx,:]
            
    fitnest_history_0.append(min(fitness_Pbest))
    fitnest_history_1.append(np.mean(fitness_Pbest))
    fitnest_history_0 = np.array(fitnest_history_0)
    fitnest_history_1 = np.array(fitnest_history_1)
    fitnest_history = np.hstack((fitnest_history_0,fitnest_history_1))
    ll = float(len(fitnest_history))/2
    fitnest_history = fitnest_history.reshape(int(ll),2,order='F')
    
    rec_1 = []
    
    for i in range(len(mape_record)):
        if mape_record[i] < 1000 and mape_record[i]>min(mape_record):
            rec_1.append(mape_record[i])

    for i in range(len(rec_1)):
        os.remove(f'pre_train_{rec_1[i]}.h5')
        os.remove(f'pre_train_{rec_1[i]}.json')
    
    model_mape = model_MAPE(mape_record, Gbest)
    
    return fitnest_history, Gbest, model_mape

def plots_():

    len_t = len(fitnest_history)
    var_x = np.zeros(len_t)
    for i in range(len_t):
        var_x[i] = i
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Minimum_fitness', color=color)
    ax1.plot(var_x,fitnest_history[:,0], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    
    color = 'tab:blue'
    ax2.set_ylabel('Average_fitness', color=color)
    # ax1.set_ylabel('Fitness', color=color)
    ax2.plot(var_x,fitnest_history[:,1], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

#Main code
fitnest_history, Gbest, model_mape = pso_opt(5,10) #N and max iteration

print(f'Minimum fitness = {min(fitnest_history[:,0])}')

plots_()
