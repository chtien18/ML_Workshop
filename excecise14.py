import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import math 

def my_function(x,y):
    z = (x-3.14)**2 + (y-2.72)**2 + math.sin(3*x+1.41) + math.sin(4*y-1.73) 
    return z

def my_function1(x1,x2):

    y = 4*x1*x1 -2.1*x1**4 +1/3*x1**6 +x1*x2 - 4*x2*x2+4*x2**4
    return y

def position_initialization(N):
    n_var = 2  # number of variable (x and y)
    initial_position = np.zeros((N,n_var))
    x1_min = -5#0
    x1_max = 5#6
    x2_min = -5#0
    x2_max = 5#6

    for i in range(N):
        initial_position[i,0] = x1_min + random.uniform(0, 1)*(x1_max-x1_min)
        initial_position[i,1] = x2_min + random.uniform(0, 1)*(x2_max-x2_min)
            
    return initial_position

def fitness_calculation(X):
    N = len(X)
    fitness_ = np.zeros(N)
    
    for i in range(N):
        x1 = X[i,0]
        x2 = X[i,1]
        fitness_[i] = my_function1(x1,x2) 
    return fitness_

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
    x1_min = 0
    x1_max = 6
    x2_min = 0
    x2_max = 6
    
    for myRow in range(N):
        if X[myRow,0] < x1_min:
            X[myRow,0] =  x1_min + random.uniform(0, 1)*(x1_max-x1_min)
        elif X[myRow,0] > x1_max:
            X[myRow,0] = x1_min + random.uniform(0, 1)*(x1_max-x1_min)
            
        if X[myRow,1] < x2_min:
            X[myRow,1] =  x2_min + random.uniform(0, 1)*(x2_max-x2_min)
        elif X[myRow,1] > x2_max:
            X[myRow,1] = x2_min + random.uniform(0, 1)*(x2_max-x2_min)
    return X

def pso_opt(N,Max_iter):  # N = number of population, Max_iter = maximum number of iteration
    
    n_var = 2  # number of variable
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
        
        fitness_new = fitness_calculation(X)
        
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

        if abs(np.mean(fitness_Pbest)-min(fitness_Pbest)) < 0.0001: #convergent criterion
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
        
        I+=1
        
    fitness_new = fitness_calculation(X)
        
    for i in range(N):
        if fitness_new[i] < fitness_Pbest[i]:
            Pbest[i,:] = copy.deepcopy(X[i,:])
            fitness_Pbest[i] = copy.deepcopy(fitness_new[i])
        else:
            Pbest[i,:] = copy.deepcopy(Pbest[i,:])
            fitness_Pbest[i] = copy.deepcopy(fitness_Pbest[i])
            
    fitnest_history_0.append(min(fitness_Pbest))
    fitnest_history_1.append(np.mean(fitness_Pbest))
    fitnest_history_0 = np.array(fitnest_history_0)
    fitnest_history_1 = np.array(fitnest_history_1)
    fitnest_history = np.hstack((fitnest_history_0,fitnest_history_1))
    ll = float(len(fitnest_history))/2
    fitnest_history = fitnest_history.reshape(int(ll),2,order='F')
    
    return fitnest_history, Gbest
    
fitnest_history, Gbest = pso_opt(50,1000)

print(f'Minimum y = {min(fitnest_history[:,0])}')
print(f'x1_best = {Gbest[0]}')
#print(f'x2_best = {Gbest[1]}')

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