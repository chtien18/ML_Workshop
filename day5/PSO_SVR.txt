import pandas as pd
import numpy as np
import copy
import random
import pickle
import matplotlib.pyplot as plt

def input_normalization(dataset_input):  # dataset_input = input parameter (PSO population/particle positions)
    dataset = pd.read_excel('Machining data.xlsx')   # dataset from experiment
    input_ = dataset.iloc[:,0:3]
    input_ = input_.values
    
    row,col = np.shape(dataset_input)
    norm_input = np.zeros((row,col))
    
    for i in range(row):
        for j in range(col):
            norm_input[i,j] = (dataset_input[i,j]-min(input_[:,j])*0.9)/(max(input_[:,j])*1.1-min(input_[:,j])*0.9)

    return norm_input

def input_denormalization(dataset_):  # dataset_input = input parameter (PSO population/particle positions)
    dataset = pd.read_excel('Machining data.xlsx')   # dataset from experiment
    input_ = dataset.iloc[:,0:3]
    input_ = input_.values
    
    row,col = np.shape(dataset_)
    denorm_ = np.zeros((row,col))
    
    for i in range(row):
        for j in range(col):
            denorm_[i,j] = min(input_[:,j])*0.9 + (max(input_[:,j])*1.1-min(input_[:,j])*0.9)*dataset_[i,j]

    return denorm_

def output_denormalization(dataset_output):
    dataset = pd.read_excel('Machining data.xlsx')   # dataset from experiment
    target_ = dataset.iloc[:,3]
    target_ = target_.values
    denorm_output = np.zeros(len(dataset_output))
                                                                
    for i in range(len(dataset_output)):
        denorm_output[i] = dataset_output[i]*max(target_)*1.05
    
    return denorm_output

'''  ---------------  PSO -------------------'''

def position_initialization(N):
    dataset = pd.read_excel('Machining data.xlsx')   # dataset from experiment
    input_ = dataset.iloc[:,0:3]
    input_ = input_.values
    
    depth_of_cut_max = max(input_[:,0])*1.1
    depth_of_cut_min = min(input_[:,0])*0.9
    feed_rate_max = max(input_[:,1])*1.1
    feed_rate_min = min(input_[:,1])*0.9
    insert_radius_max = max(input_[:,2])*1.1
    insert_radius_min = min(input_[:,2])*0.9
    
    initial_position = np.zeros((N,3))

    for i in range(N):
        initial_position[i,0] = depth_of_cut_min + random.uniform(0, 1)*(depth_of_cut_max-depth_of_cut_min)
        initial_position[i,1] = feed_rate_min + random.uniform(0, 1)*(feed_rate_max-feed_rate_min)
        initial_position[i,2] = insert_radius_min + random.uniform(0, 1)*(insert_radius_max-insert_radius_min)
        
    return initial_position

def objective_function(X):           # X is normalized inputs
    filename = 'SVR_model_fold3.sav'
    # load the model from disk
    model = pickle.load(open(filename, 'rb'))
    y_pred = model.predict(X)
    
    return y_pred

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
    
    dataset = pd.read_excel('Machining data.xlsx')   # dataset from experiment
    input_ = dataset.iloc[:,0:3]
    input_ = input_.values
    
    # Input normalization
    
    norm_input = input_normalization(input_)
    
    depth_of_cut_max = max(norm_input[:,0]*1.1)
    depth_of_cut_min = min(norm_input[:,0]*0.9)
    feed_rate_max = max(norm_input[:,1]*1.1)
    feed_rate_min = min(norm_input[:,1]*0.9)
    insert_radius_max = max(norm_input[:,2]*1.1)
    insert_radius_min = min(norm_input[:,2]*0.9)
    
    for i in range(N):
        if X[i,0] < depth_of_cut_min:
            X[i,0] = depth_of_cut_min + random.uniform(0, 1)*(depth_of_cut_max-depth_of_cut_min)
        elif X[i,0] > depth_of_cut_max:
            X[i,0] = depth_of_cut_min + random.uniform(0, 1)*(depth_of_cut_max-depth_of_cut_min)
        
        if X[i,1] < feed_rate_min:
            X[i,1] = feed_rate_min + random.uniform(0, 1)*(feed_rate_max-feed_rate_min)
        elif X[i,1] > feed_rate_max:
            X[i,1] = feed_rate_min + random.uniform(0, 1)*(feed_rate_max-feed_rate_min)
        
        if X[i,2] < insert_radius_min:
            X[i,2] = insert_radius_min + random.uniform(0, 1)*(insert_radius_max-insert_radius_min)
        elif X[i,2] > insert_radius_max:
            X[i,2] = insert_radius_min + random.uniform(0, 1)*(insert_radius_max-insert_radius_min)

    return X

def pso_opt(N,Max_iter):
    
    n_var = 3
    fitnest_history_0 = []
    fitnest_history_1 = []
    
    X_0 = position_initialization(N)   # Initial positions before normalization
    X = input_normalization(X_0)      # Don't forget to normalize the inputs
    Xnew = np.zeros((N,n_var))          
    V = 0.1*X                        # Initial velocity
    Vnew = np.zeros((N,n_var))
    
    c1 = 2
    c2 = 2
    
    I = 0

    while I < Max_iter:
        
        fitness_new = objective_function(X)   # Calculate the fitness values
        
        if I ==0:
            X = X
            V = V
            Pbest = copy.deepcopy(X)
            fitness_Pbest = copy.deepcopy(fitness_new)
            idx = find_index(fitness_Pbest)
            Gbest = Pbest[idx,:]
        
        else:
            for i in range(N):               # memory saving
                if fitness_new[i] < fitness_Pbest[i]:
                    Pbest[i,:] = copy.deepcopy(X[i,:])
                    fitness_Pbest[i] = copy.deepcopy(fitness_new[i])
                else:
                    Pbest[i,:] = copy.deepcopy(Pbest[i,:])
                    fitness_Pbest[i] = copy.deepcopy(fitness_Pbest[i])
        
        idx = find_index(fitness_Pbest)     # Find the index for the smallest fitness
            
        Gbest = Pbest[idx,:]		    # Gbest is the best particles in population
        
        fitnest_history_0.append(min(fitness_Pbest))
        fitnest_history_1.append(np.mean(fitness_Pbest))
        print(f'Iteration {I}')
        print(f'minimum fitness: {min(fitness_Pbest)}')
        print(f'average fitness: {np.mean(fitness_Pbest)}')

        if abs(np.mean(fitness_Pbest)-min(fitness_Pbest)) < 0.0001: #convergent criterion
            break

        r1 = np.zeros((N,n_var))
        r2 = np.zeros((N,n_var))
        for i in range(N):
            for j in range(n_var):
                r1[i,j] = random.uniform(0,1)
                r2[i,j] = random.uniform(0,1)
                
        Gbest = Gbest.reshape(1, -1)

        for i in range(N):              # Updating rates and positions of particles
            for j in range(n_var):
                w_max = 0.9
                w_min = 0.4
                w = (w_max-w_min)*(Max_iter-I)/Max_iter + w_min
                Vnew[i,j] = w*V[i,j] + c1*r1[i,j]*(Pbest[i,j]-X[i,j]) + c2*r2[i,j]*(Gbest[0,j]-X[i,j])
                Xnew[i,j] = X[i,j] + Vnew[i,j]
        
        X = copy.deepcopy(Xnew)
        V = copy.deepcopy(Vnew)
        
        X = boundary_handling(X)
        
        I+=1
    
    fitness_new = objective_function(X)     # Calculate the fitness values of the final generation
    
    for i in range(N):                        # Memory saving
        if fitness_new[i] < fitness_Pbest[i]:
            Pbest[i,:] = copy.deepcopy(X[i,:])
            fitness_Pbest[i] = copy.deepcopy(fitness_new[i])
        else:
            Pbest[i,:] = copy.deepcopy(Pbest[i,:])
            fitness_Pbest[i] = copy.deepcopy(fitness_Pbest[i])
            
    idx = find_index(fitness_Pbest)       # Find the index for the smallest fitness
    
    Pbest_final = input_denormalization(Pbest)             # Don't forget to denormalize the inputs (process parameters)
    best_output = output_denormalization(fitness_Pbest)    # Don't forget to denormalize the output (surface roughness)
    
    Gbest_final = Pbest_final[idx,:]             # This is the optimal combination of process parameters
    best_output_final = best_output[idx]         # This is the optimal surface roughness
            
    fitnest_history_0.append(min(fitness_Pbest))
    fitnest_history_1.append(np.mean(fitness_Pbest))
    fitnest_history_0 = np.array(fitnest_history_0)
    fitnest_history_1 = np.array(fitnest_history_1)
    fitnest_history = np.hstack((fitnest_history_0,fitnest_history_1))
    ll = float(len(fitnest_history))/2
    fitnest_history = fitnest_history.reshape(int(ll),2,order='F')
    
    return fitnest_history, Gbest_final, best_output_final

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


fitnest_history, Gbest_final, best_output_final = pso_opt(50,1000)

print('')
print('Final report:')
print(f'surface_roughness_optimum = {best_output_final}')
print(f'depth_of_cut_optimum = {Gbest_final[0]}')
print(f'feed_rate_optimum = {Gbest_final[1]}')
print(f'insert_radius_optimum = {Gbest_final[2]}')

plots_()