import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import copy
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import KFold
import pickle
import math
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import statistics

def myround(x, prec=0, base=1):
    return round(base * round(float(x)/base),prec)

def myround_3(x, prec=3, base=0.001):
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

    return x, y

def SVR_(C,epsilon,RSN):
    arr_mape_cv = np.zeros((5,2))
    x, y = dataset_prep(RSN)
    
    NK = 5       # Number of fold
    kf = KFold(NK, shuffle=False) 
    model = SVR(kernel = 'rbf',C=C, epsilon=epsilon)
    fold = 0                                        # We use 5-fold cross validation
    for train, test in kf.split(x):
        fold+=1
        
        x_train = copy.deepcopy(x[train])
        y_train = copy.deepcopy(y[train])
        x_test = copy.deepcopy(x[test])
        y_test = copy.deepcopy(y[test])

        model.fit(x_train, y_train)
        
        y_hat_train = model.predict(x_train)
        y_hat_test = model.predict(x_test)

        mape_train = mean_absolute_percentage_error(y_train,y_hat_train)*100
        mape_test = mean_absolute_percentage_error(y_test,y_hat_test)*100

        arr_mape_cv[fold-1,0] = copy.deepcopy(mape_train)
        arr_mape_cv[fold-1,1] = copy.deepcopy(mape_test)

    f1 = np.mean(arr_mape_cv[:,0])
    f2 = np.mean(arr_mape_cv[:,1])

    fitness_ = (f1+f2)/2

    return fitness_

'''  ---------------  PSO -------------------'''

def position_initialization(N):
    initial_position = np.zeros((N,3))
    C_min = 0
    C_max = 100
    epsilon_min = 0
    epsilon_max = 1
    RSN_min = 0
    RSN_max = 100

    for i in range(N):
        initial_position[i,0] = myround_3(C_min + random.uniform(0, 1)*(C_max-C_min))
        initial_position[i,1] = myround_3(epsilon_min + random.uniform(0, 1)*(epsilon_max-epsilon_min))
        initial_position[i,2] = myround(RSN_min + random.uniform(0, 1)*(RSN_max-RSN_min))
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
    C_min = 0
    C_max = 100
    epsilon_min = 0
    epsilon_max = 1
    RSN_min = 0
    RSN_max = 100

    for i in range(N):
        if X[i,0] < C_min:
            X[i,0] = myround_3(C_min + random.uniform(0, 1)*(C_max-C_min))
        elif X[i,0] > C_max:
            X[i,0] = myround_3(C_min + random.uniform(0, 1)*(C_max-C_min))
        
        if X[i,1] < epsilon_min:
            X[i,1] = myround_3(epsilon_min + random.uniform(0, 1)*(epsilon_max-epsilon_min))
        elif X[i,1] > epsilon_max:
            X[i,1] = myround_3(epsilon_min + random.uniform(0, 1)*(epsilon_max-epsilon_min))
        
        if X[i,2] < RSN_min:
            X[i,2] = myround(RSN_min + random.uniform(0, 1)*(RSN_max-RSN_min))
        elif X[i,2] > RSN_max:
            X[i,2] = myround(RSN_min + random.uniform(0, 1)*(RSN_max-RSN_min))

    return X

def pso_opt(N,Max_iter):
    
    n_var = 3
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
            C = X[i,0]
            epsilon = X[i,1]
            RSN = int(X[i,2])
            
            fitness_new[i] = SVR_(C,epsilon,RSN)
        
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
            X[i,0] = myround_3(X[i,0])
            X[i,1] = myround_3(X[i,1])
            X[i,2] = myround(X[i,2])
        I+=1
        
    fitness_new = np.zeros(len(X))
    for i in range(len(X)):
        C = X[i,0]
        epsilon = X[i,1]
        RSN = X[i,2]
        
        fitness_new[i] = SVR_(C,epsilon,RSN)
        
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
    
    return fitnest_history, Gbest

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

def SVR_final(C,epsilon,RSN):
    arr_mape_cv = np.zeros((5,2))
    x, y = dataset_prep(RSN)
    
    NK = 5       # Number of fold
    kf = KFold(NK, shuffle=False) 
    model = SVR(kernel = 'rbf',C=C, epsilon=epsilon)
    fold = 0                                        # We use 5-fold cross validation
    for train, test in kf.split(x):
        fold+=1
        
        x_train = copy.deepcopy(x[train])
        y_train = copy.deepcopy(y[train])
        x_test = copy.deepcopy(x[test])
        y_test = copy.deepcopy(y[test])

        model.fit(x_train, y_train)
        
        filename = f'SVR_model_fold{fold}.sav'
        pickle.dump(model, open(filename, 'wb'))
        
        y_hat_train = model.predict(x_train)
        y_hat_test = model.predict(x_test)

        mape_train = mean_absolute_percentage_error(y_train,y_hat_train)*100
        mape_test = mean_absolute_percentage_error(y_test,y_hat_test)*100

        arr_mape_cv[fold-1,0] = copy.deepcopy(mape_train)
        arr_mape_cv[fold-1,1] = copy.deepcopy(mape_test)

    return arr_mape_cv

fitnest_history, Gbest = pso_opt(50,100)

C = Gbest[0]
epsilon = Gbest[1]
RSN = int(Gbest[2])

arr_mape_cv = SVR_final(C,epsilon,RSN)
arr_mape_cv = pd.DataFrame(arr_mape_cv)
arr_mape_cv.columns = ['MAPE training', 'MAPE testing']

print(f'Minimum fitness = {min(fitnest_history[:,0])}')
print(f'C_best = {Gbest[0]}')
print(f'epsilon_best = {Gbest[1]}')
print(f'RSN_best = {int(Gbest[2])}')

plots_()