from math import ceil
import numpy as np 
import cvxpy as cp 
import random
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank

if __name__ == '__main__':
    # Define system
    A = np.array([[1.01,0.01,0],[0.01,1.01,0.01],[0,0.01,1.01]])
    B = np.eye(3)
    C = np.eye(3)
    (n,m) = B.shape                             # n = number of states, m = number of inputs
    p = n                                       # p = number of output
    
    # Cost weight
    Q = np.eye(3)
    R = np.eye(3)*1000
       
    # Simulation    
    x0 = np.array([[0],[0],[1]])
    t_step = 0.1
    t_sim = 5.1;                                   # simulation time
    n_sim = ceil(t_sim/t_step)                     # Number of simulation step
    x = x0; 
    xLog = np.empty((n,n_sim+1)); 
    xLog[:,[0]] = x0; 
    uLog = np.empty((m,n_sim));
    
    # Generate random input sequence
    np.random.seed(1)
    u_seq = np.array(np.random.rand(m,n_sim))

    # Collect data
    for t in range(n_sim):
        u = u_seq[:, t].reshape(m, 1)
        x = A@x+B@u
        uLog[:,[t]] = u 
        xLog[:,[t+1]] = x
    
    # Create Hankel_U matrix
    # condition: T - T_f + 1 >= T_f*m
    T_ini = 3
    N = 5
    T_L = T_ini + N 
    nB = 3
    T_f = T_ini + N + nB #13
    T = n_sim #51    51 - 8 + 1 = 44
    Hankel_U = np.empty((T_L*m,T-T_L+1))
    Hankel_Y = np.empty((T_L*p,T-T_L+1))
    for i in range(T_L):
        Hankel_U[m*i:m*(i+1),:] = u_seq[:,i:T-T_L+i+1]
        Hankel_Y[p*i:p*(i+1),:] = xLog[:,i+1:T-T_L+i+2]
    
    # Construct U_p, U_f, Y_p, Y_f    
    U_p = Hankel_U[:m*T_ini,:] 
    U_f = Hankel_U[m*T_ini:,:]

    Y_p = Hankel_Y[:m*T_ini,:]
    Y_f = Hankel_Y[m*T_ini:,:]
    
    Hankel_PF = np.block([[U_p],[Y_p],[U_f],[Y_f]])
    # print(U_p.shape)
    # print(Hankel_PF.shape)
    
    # Construct ini_matrix
    ini_matrix = np.empty((n*m+n*p,1))
    ini_matrix[:n*m,:] = np.reshape(u_seq[:,-n:],(-1, 1), order='F')           # last n element
    ini_matrix[n*m:n*m+n*p,:] = np.reshape(xLog[:,-n:],(-1, 1), order='F')     # last n element
    print(ini_matrix.shape)
        
    # BB = np.block([[u_seq[:,-1].reshape(m, 1)],[u_seq[:,-1].reshape(m, 1)]])
    # BB = np.block([[BB],[BB]])
    # print(BB)
    
    # Set up DeePC problem
    

    
    
        
    # Plot system evolution
    # time = np.linspace(0,t_step,n_sim+1)
    # plt.plot(time,xLog[2,:])
    # plt.show()